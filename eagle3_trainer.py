import os
import logging
import json
import gc
import deepspeed
import argparse

from safetensors.torch import safe_open
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model.llama_eagle3 import LlamaForCausalLMEagle3
from utils import Tracking

set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger(__file__)


def add_args():
    parser = argparse.ArgumentParser(description="Eagle3 Trainer DeepSpeed")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument(
        "--freq_map_path", type=str, default="freq_map/llama3/freq_32768.pt", help="Path to frequency mapping file"
    )
    parser.add_argument("--draft_vocab_size", type=int, default=32768, help="Draft vocabulary size")
    parser.add_argument("--prediction_length", type=int, default=7, help="Number of prediction steps")
    parser.add_argument("--project_name", type=str, default="Eagle3-Trainer", help="WandB project name")
    parser.add_argument("--experiment_name", type=str, default="eagle3-deepspeed", help="WandB experiment name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss")
    parser.add_argument("--prob_weight", type=float, default=0.1, help="Weight for probability loss")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--load_optimizer", action="store_true", help="Whether to load optimizer states")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"], help="Training precision type")

    # Online target model inference support
    parser.add_argument("--use_target_model", action="store_true", help="Use target model to generate hidden states")
    parser.add_argument(
        "--target_model_layers", type=str, default="0,1,2", help="Comma-separated list of layer indices to extract"
    )

    parser = deepspeed.add_config_arguments(parser)
    return parser


class EagleDataset(Dataset):

    def __init__(
        self,
        datapath,
        transform=None,
        max_len=2048,
        dataset_max_len=None,
        hidden_states_idx=None,
        target_model=None,
        tokenizer=None,
        use_target_model=False,
        target_model_layers=[0, 1, 2],
    ):
        """Initialize EagleDataset to load pre-processed data or generate hidden states dynamically"""
        self.datapath = datapath
        self.transform = transform
        self.max_len = max_len
        self.global_max_seq_len = dataset_max_len
        self.hidden_states_idx = hidden_states_idx
        self.failed_indices = set()  # Track failed loads

        # Target model support
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.use_target_model = use_target_model
        self.target_model_layers = target_model_layers

        if self.use_target_model and (self.target_model is None or self.tokenizer is None):
            raise ValueError("target_model and tokenizer must be provided when use_target_model=True")

    def __len__(self):
        return len(self.datapath)

    @torch.no_grad()
    def _generate_hidden_states(self, input_ids, attention_mask):
        """Generate hidden states using target model"""
        if not self.use_target_model:
            return None

        # Ensure tensors are on the same device as target model
        device = next(self.target_model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Generate outputs with hidden states
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                attention_mask=attention_mask.unsqueeze(0),
                output_hidden_states=True,
            )

        # Extract hidden states from specified layers
        hidden_states_list = []
        for layer_idx in self.target_model_layers:
            if layer_idx < len(outputs.hidden_states):
                hidden_states_list.append(outputs.hidden_states[layer_idx])

        # Concatenate hidden states from different layers
        if len(hidden_states_list) > 1:
            hidden_states = torch.cat(hidden_states_list, dim=-1)
        else:
            hidden_states = hidden_states_list[0]

        return hidden_states.squeeze(0)  # Remove batch dimension

    def _padding_right(self, tensor, target_length):
        """Pad tensor to target length on the right"""
        current_length = tensor.shape[0]
        if current_length >= target_length:
            return tensor[:target_length]

        padding_length = target_length - current_length
        if len(tensor.shape) == 1:
            padding = torch.zeros(padding_length, dtype=tensor.dtype, device=tensor.device)
        else:
            padding_shape = [padding_length] + list(tensor.shape[1:])
            padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)

        return torch.cat([tensor, padding], dim=0)

    def __getitem__(self, idx):
        if idx in self.failed_indices:
            return self.__getitem__((idx + 1) % len(self.datapath))

        try:
            if self.use_target_model:
                # Load raw text data for dynamic hidden state generation
                # Assuming the data files contain text or input_ids
                if self.datapath[idx].endswith(".json"):
                    with open(self.datapath[idx], "r") as f:
                        data = json.load(f)

                    # Expect format: {"text": "...", "input_ids": [...], "attention_mask": [...]}
                    if "input_ids" in data:
                        input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
                        attention_mask = torch.tensor(data.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
                    else:
                        # Tokenize text if only text is provided
                        text = data.get("text", "")
                        encoded = self.tokenizer(text, return_tensors="pt", max_length=self.max_len, truncation=True)
                        input_ids = encoded["input_ids"].squeeze(0)
                        attention_mask = encoded["attention_mask"].squeeze(0)

                    # Generate loss mask (assuming all tokens contribute to loss for now)
                    loss_mask = torch.ones_like(input_ids, dtype=torch.float)

                    # Generate hidden states using target model
                    hidden_states = self._generate_hidden_states(input_ids, attention_mask)

                else:
                    # For .pt files, assume they contain input_ids and possibly other data
                    data = torch.load(self.datapath[idx], weights_only=True)
                    input_ids = data["input_ids"]
                    attention_mask = data.get("attention_mask", torch.ones_like(input_ids))
                    loss_mask = data.get("loss_mask", torch.ones_like(input_ids, dtype=torch.float))

                    # Generate hidden states using target model
                    hidden_states = self._generate_hidden_states(input_ids, attention_mask)

                # Prepare target hidden states (shift right by 1 for next-token prediction)
                target_hidden_states = hidden_states[1:] if len(hidden_states) > 1 else hidden_states

                processed_item = {
                    "input_ids": input_ids[1:],  # Shift right by 1
                    "hidden_states": hidden_states[:-1] if len(hidden_states) > 1 else hidden_states,  # Current hidden states
                    "target": target_hidden_states,  # Target hidden states (shifted)
                    "loss_mask": loss_mask[1:] if len(loss_mask) > 1 else loss_mask,  # Shift loss mask
                    "max_seq_len": self.global_max_seq_len,
                }

            else:
                # Original behavior: load pre-computed hidden states from files
                data = torch.load(self.datapath[idx], weights_only=True)
                data["loss_mask"][-1] = 0
                all_hidden_states = torch.cat(list(data["hidden_states_dict"].values()), dim=0)
                hidden_states = all_hidden_states[:-1, :, :]
                last_hidden_states = all_hidden_states[-1, :, :]
                processed_item = {
                    "input_ids": data["input_ids"][1:],
                    "hidden_states": hidden_states,
                    "target": hidden_states[:, 1:, :],  # Shift right by 1 for target
                    "last_hidden_states": last_hidden_states,
                    "loss_mask": data["loss_mask"],
                    "max_seq_len": self.global_max_seq_len,
                }

            if self.transform:
                processed_item = self.transform(processed_item)

            return processed_item

        except Exception as e:
            self.failed_indices.add(idx)
            if dist.get_rank() == 0:
                logger.warning(f"Error loading file {self.datapath[idx]}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self.datapath))


class EagleDataCollator:

    def padding_tensor(self, input, N):
        input = input.unsqueeze(0)

        if input.dim() < 4:
            B, n = input.shape[:2]
        else:
            assert input.shape[1] == 3, f"got {input.shape}"  # Eagle-3 uses 3 layers' hidden_states [1, 3, n, d]
            B, _, n = input.shape[:3]

        padding_length = N - n
        if len(input.shape) == 2:  # [B, n]
            output = torch.nn.functional.pad(input, (0, padding_length), value=0)
        elif len(input.shape) >= 3:  # [B, n, d] / [B, 3, n, d] for hidden states
            output = torch.nn.functional.pad(input, (0, 0, 0, padding_length), value=0)
        else:
            raise ValueError(f"Unsupported tensor shape: {input.shape}")
        return output

        # padding_shape = [B, N - n]
        # if len(input.shape) > 2:
        #     padding_shape.append(input.shape[2])
        # padding_tensor = torch.zeros(padding_shape, dtype=input.dtype, device=input.device)
        # output = torch.cat((input, padding_tensor), dim=1)
        # return output

    def __call__(self, features):
        # Determine max length from the batch if max_seq_len is not set
        max_length = features[0]["max_seq_len"]
        if max_length is None:
            max_length = max(len(item["input_ids"]) for item in features)
            logger.warning(f"max_seq_len not set in dataset, using batch maximum length: {max_length}")

        device = features[0]["input_ids"].device

        batch_attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.ones(len(item["input_ids"]) + 1, device=device),
                        torch.zeros(max_length - len(item["input_ids"]) - 1, device=device),
                    ]
                ).unsqueeze(0)
                for item in features
            ]
        )
        batch_input_ids = torch.cat([self.padding_tensor(item["input_ids"], max_length) for item in features])
        batch_hidden_states = torch.cat([self.padding_tensor(item["hidden_states"], max_length) for item in features])
        batch_target = torch.cat([self.padding_tensor(item["target"], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [
                torch.cat([item["loss_mask"], torch.zeros(max_length - len(item["loss_mask"]), device=device)]).unsqueeze(0)
                for item in features
            ]
        )

        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "loss_mask": batch_loss_mask,
            "attention_mask": batch_attention_mask,
        }

        # Only include last_hidden_states if present (not used when use_target_model=True)
        if "last_hidden_states" in features[0]:
            batch_last_hidden_states = torch.cat(
                [self.padding_tensor(item["last_hidden_states"], max_length) for item in features]
            )
            batch["last_hidden_states"] = batch_last_hidden_states

        return batch


class Eagle3TrainerDeepSpeed:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.start_epoch = 0  # Track starting epoch for resume
        self.prediction_length = args.prediction_length

        # Load vocabulary mapping
        self._load_vocab_mapping()

        # Load target model if specified
        self.target_model = None
        self.tokenizer = None
        if args.use_target_model:
            self._load_target_model()

        # Parse target model layers
        if args.target_model_layers:
            self.target_model_layers = [int(x.strip()) for x in args.target_model_layers.split(",")]
        else:
            self.target_model_layers = [0, 1, 2]

        # Initialize model and datasets
        self._build_dataloader()
        self._build_model()
        self._initialize_deepspeed()

        self._load_checkpoint()

        # Initialize loss functions
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def _load_target_model(self):
        """Load target model for dynamic hidden state generation"""
        if self.rank == 0:
            logger.info(f"Loading target model from: {self.args.base_model_path}")
            logger.info(f"Target model layers: {self.target_model_layers}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load target model
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model_path, torch_dtype=dtype, device_map=f"cuda:{self.local_rank}", trust_remote_code=True
        )

        # Set model to eval mode and freeze parameters
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        if self.rank == 0:
            logger.info(f"Target model loaded successfully on device: cuda:{self.local_rank}")
            logger.info(f"Target model vocabulary size: {self.target_model.config.vocab_size}")

    def _load_vocab_mapping(self):
        """Load pre-computed vocabulary mapping"""
        if not os.path.exists(self.args.freq_map_path):
            if self.rank == 0:
                logger.warning(f"Vocabulary mapping file not found: {self.args.freq_map_path}")

        mapping_data = torch.load(self.args.freq_map_path, map_location="cpu")
        self.d2t = mapping_data.get("d2t", None)
        self.t2d = mapping_data.get("t2d", None)

        if self.d2t is None or self.t2d is None:
            raise ValueError(f"Vocabulary mapping not found in {self.args.freq_map_path}, generate it first")

        if self.rank == 0:
            logger.info(f"Loaded vocabulary mapping: draft_vocab_size={len(self.d2t)}, vocab_size={len(self.t2d)}")
            logger.info(f"t2d max value: {torch.max(self.t2d).item()}, min value: {torch.min(self.t2d).item()}")
            logger.info(f"d2t max value: {torch.max(self.d2t).item()}, min value: {torch.min(self.d2t).item()}")

    def _load_checkpoint(self):
        if os.path.exists(os.path.join(self.args.output_dir, "latest")):
            load_path, client_state = self.model_engine.load_checkpoint(
                self.args.output_dir,
                load_optimizer_states=self.args.load_optimizer,
                load_lr_scheduler_states=True,
                load_module_only=False,
            )

            self.start_epoch = client_state["epoch"] + 1
            steps_per_epoch = len(self.train_loader)
            self.start_epoch = client_state["step"] // steps_per_epoch + 1

            if self.rank == 0:
                logger.info(f"Successfully loaded checkpoint from: {load_path}")
                logger.info(f"Resuming training from epoch {self.start_epoch}")
                if client_state:
                    logger.info(f"Client state keys: {list(client_state.keys())}")

    def _build_model(self):
        # Load model config
        config = AutoConfig.from_pretrained(self.args.base_model_path, trust_remote_code=True)
        config.num_hidden_layers = 1
        config.torch_dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16
        config._attn_implementation = "flash_attention_2"

        # Add Eagle3 specific config
        config.draft_vocab_size = self.args.draft_vocab_size

        # Configure target hidden size based on target model layers
        if self.args.use_target_model and self.target_model:
            target_hidden_size = self.target_model.config.hidden_size * len(self.target_model_layers)
            config.target_hidden_size = target_hidden_size
            if self.rank == 0:
                logger.info(f"Target hidden size: {target_hidden_size} (layers: {self.target_model_layers})")

        # Determine model class - only support Llama for Eagle3
        if hasattr(config, "model_type"):
            if config.model_type.lower() == "llama":
                model_class = LlamaForCausalLMEagle3
            # elif config.model_type.lower() == "qwen2":
            #     model_class = Qwen2ForCausalLMEagle
            else:
                raise ValueError(f"Eagle3 currently only supports Llama models, got: {config.model_type}")
        else:
            if "llama" in self.args.base_model_path.lower():
                model_class = LlamaForCausalLMEagle3
            else:
                raise ValueError("Eagle3 currently only supports Llama and Qwen models")

        self.model = model_class(config=config)

        # Register vocabulary mapping buffers
        self.model.register_buffer("d2t", self.d2t)
        self.model.register_buffer("t2d", self.t2d)

        # Precompute valid t2d mapping for target model if using target model
        if self.args.use_target_model and self.target_model:
            target_vocab_size = self.target_model.config.vocab_size
            if torch.any(self.t2d >= target_vocab_size):
                if self.rank == 0:
                    logger.warning(f"t2d contains indices >= target vocab size ({target_vocab_size}). Clamping to valid range.")
                valid_t2d = torch.clamp(self.t2d, 0, target_vocab_size - 1)
                self.model.register_buffer("valid_t2d", valid_t2d)
            else:
                self.model.register_buffer("valid_t2d", self.t2d)
        else:
            self.model.register_buffer("valid_t2d", self.t2d)

        # Load embeddings and LM head from base model
        self._load_base_model_weights()
        self.model.to(dtype=config.torch_dtype)

    def _load_base_model_weights(self):
        base_path = self.args.base_model_path
        device = f"cuda:{self.local_rank}"
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float16

        try:
            # Try loading from safetensors first
            with open(os.path.join(base_path, "model.safetensors.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]

            with safe_open(os.path.join(base_path, head_path), framework="pt", device=device) as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                lm_head_weight = tensor_slice[:, :hidden_dim].to(dtype)

            with safe_open(os.path.join(base_path, embed_path), framework="pt", device=device) as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                embed_weight = tensor_slice[:, :hidden_dim].to(dtype)
        except:
            # Fallback to pytorch model files
            with open(os.path.join(base_path, "pytorch_model.bin.index.json"), "r") as f:
                index_json = json.loads(f.read())
                head_path = index_json["weight_map"]["lm_head.weight"]
                embed_path = index_json["weight_map"]["model.embed_tokens.weight"]

            head_weights = torch.load(os.path.join(base_path, head_path))
            embed_weights = torch.load(os.path.join(base_path, embed_path))
            lm_head_weight = head_weights["lm_head.weight"].to(dtype).to(device)
            embed_weight = embed_weights["model.embed_tokens.weight"].to(dtype).to(device)

        # Copy weights to model (embedding tokens only, not LM head as it has different vocab size)
        self.model.model.embed_tokens.weight.data.copy_(embed_weight)

        # For Eagle3, initialize LM head with subset of original LM head weights
        target_vocab_indices = self.model.d2t
        if len(target_vocab_indices) <= lm_head_weight.shape[0]:
            mapped_lm_head_weight = lm_head_weight[target_vocab_indices]
            self.model.lm_head.weight.data.copy_(mapped_lm_head_weight)
        else:
            # If mapping is larger than available vocab, use available portion
            available_size = min(len(target_vocab_indices), lm_head_weight.shape[0])
            self.model.lm_head.weight.data[:available_size].copy_(lm_head_weight[:available_size])

        # Freeze embedding layer only (LM head needs to be trained for draft vocabulary)
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False

        if not self.args.use_target_model:
            self.base_model_lm_head = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size, bias=False)
            self.base_model_lm_head.weight.data.copy_(lm_head_weight)

            for param in self.base_model_lm_head.parameters():
                param.requires_grad = False
        else:
            self.base_model_lm_head = None

        del embed_weight, lm_head_weight
        torch.cuda.empty_cache()
        gc.collect()

    def _build_dataloader(self):
        # First, check if the data path exists
        if not os.path.exists(self.args.data_path):
            raise ValueError(f"Data path does not exist: {self.args.data_path}")

        logger.info(f"Searching for data files in: {self.args.data_path}")

        # Determine max sequence length from directory name if possible
        dataset_max_len = 8192  # Default max length

        if self.args.use_target_model:
            datapath = []
            for root, dirs, files in os.walk(self.args.data_path):
                for file in files:
                    if file.endswith((".json", ".pt", ".ckpt")):
                        file_path = os.path.join(root, file)
                        datapath.append(file_path)

            if not datapath:
                raise ValueError(f"No suitable data files found in {self.args.data_path}")

            if self.rank == 0:
                logger.info(f"Found {len(datapath)} data files for target model mode")
                logger.info("Example file paths:")
                for path in datapath[:3]:
                    logger.info(f"  {path}")
        else:
            # Original behavior: look for pre-computed hidden state files
            # List all subdirectories
            dir_list = [d for d in os.listdir(self.args.data_path) if os.path.isdir(os.path.join(self.args.data_path, d))]
            dir_list.sort()
            logger.info(f"Found directories: {dir_list}")

            for dirname in dir_list:
                if "-" in dirname:
                    try:
                        max_k_value = dirname.split("-")[1].strip()
                        if max_k_value.endswith("K"):
                            dataset_max_len = int(float(max_k_value[:-1]) * 1024)
                        else:
                            dataset_max_len = int(max_k_value)
                        break
                    except:
                        continue

            logger.info(f"Using max sequence length: {dataset_max_len}")

            datapath = []
            for root, dirs, files in os.walk(self.args.data_path):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".ckpt"):
                        file_path = os.path.join(root, file)
                        datapath.append(file_path)

            if not datapath:
                raise ValueError(f"No .ckpt/.pt files found in {self.args.data_path} or its subdirectories")

            logger.info(f"Total number of .ckpt/.pt files found: {len(datapath)}")

        # Split into train and validation
        train_size = int(len(datapath) * 0.95)
        train_files = datapath[:train_size]
        val_files = datapath[train_size:]

        # Create datasets with proper parameters
        self.train_dataset = EagleDataset(
            train_files,
            dataset_max_len=dataset_max_len,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            use_target_model=self.args.use_target_model,
            target_model_layers=self.target_model_layers,
        )
        self.val_dataset = EagleDataset(
            val_files,
            dataset_max_len=dataset_max_len,
            target_model=self.target_model,
            tokenizer=self.tokenizer,
            use_target_model=self.args.use_target_model,
            target_model_layers=self.target_model_layers,
        )

        if self.rank == 0:
            logger.info(f"Found {len(train_files)} training files and {len(val_files)} validation files")
            # Log some example file paths for verification
            logger.info("Example training file paths:")
            for path in train_files[:3]:
                logger.info(f"  {path}")
            if val_files:
                logger.info("Example validation file paths:")
                for path in val_files[:3]:
                    logger.info(f"  {path}")

        # Verify data loading by trying to load the first file
        if train_files:
            try:
                first_item = self.train_dataset[0]
                logger.info(f"Successfully loaded first data item")
                logger.info(f"Data keys: {list(first_item.keys())}")
                if "input_ids" in first_item:
                    logger.info(f"First item input_ids length: {len(first_item['input_ids'])}")
                if "hidden_states" in first_item:
                    logger.info(f"First item hidden_states shape: {first_item['hidden_states'].shape}")
            except Exception as e:
                logger.error(f"Failed to load first data item: {str(e)}")
                raise

    def _initialize_deepspeed(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize DeepSpeed engine
        self.model_engine, self.optimizer, self.train_loader, _ = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=parameters,
            training_data=self.train_dataset,
            collate_fn=EagleDataCollator(),
        )

        # Only initialize base_model_engine when not using target model
        if not self.args.use_target_model and self.base_model_lm_head is not None:
            self.base_model_engine = self.base_model_lm_head.to(dtype=self.model.config.torch_dtype).to(
                device=f"cuda:{self.local_rank}"
            )
        else:
            self.base_model_engine = None

        # Create validation dataloader
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=EagleDataCollator(), num_workers=4
        )

    @torch.no_grad()
    def _padding(self, tensor, left=True):
        """Utility function to pad tensors as used in Eagle3"""
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        else:
            tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
        return tensor

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Create causal mask for attention"""
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = torch.triu(
                torch.full(
                    (input_shape[-1], input_shape[-1]), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device
                ),
                diagonal=1,
            )[None, None, :, :]

        if attention_mask is not None:
            expanded_attn_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(inputs_embeds.dtype).min
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _compute_loss(self, batch):
        """Compute Eagle3 multi-step prediction losses"""
        input_ids = batch["input_ids"]
        hidden_states = batch["hidden_states"]  # Pre-cached hidden states from target model or generated dynamically
        last_hidden_states = batch.get("last_hidden_states", None)  # May not exist when use_target_model=True
        loss_mask = batch["loss_mask"]
        attention_mask = batch["attention_mask"]

        batch_size, seq_length = input_ids.shape

        # Prepare target logits
        with torch.no_grad():
            if self.args.use_target_model and self.target_model:
                # Generate target logits using target model dynamically
                # For target model, we need to get the full sequence including the first token
                # that was shifted out in the dataset
                # Use bos_token_id if available, otherwise use eos_token_id, otherwise use 1
                if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                    start_token_id = self.tokenizer.bos_token_id
                elif hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                    start_token_id = self.tokenizer.eos_token_id
                else:
                    start_token_id = 1  # Default fallback

                full_input_ids = torch.cat(
                    [torch.full((batch_size, 1), start_token_id, dtype=input_ids.dtype, device=input_ids.device), input_ids],
                    dim=1,
                )
                full_attention_mask = torch.cat(
                    [
                        torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[:, :-1],  # Remove last padding
                    ],
                    dim=1,
                )

                # Generate target model outputs
                target_outputs = self.target_model(
                    input_ids=full_input_ids, attention_mask=full_attention_mask, output_hidden_states=True
                )
                target_logits = target_outputs.logits
                target_logits = target_logits[:, 1:, :]  # Remove first token, align with current sequence
            else:
                # Use cached hidden states approach (original implementation)
                if last_hidden_states is None:
                    raise ValueError("last_hidden_states is required when use_target_model=False")
                last_hidden_states = self._padding(last_hidden_states, left=False)  # Shift right for target
                # Use model's embedding and lm_head to get target logits
                target_logits = self.base_model_engine(last_hidden_states.to(dtype=self.model.config.torch_dtype))

            target_max_token = target_logits.argmax(-1)

            # Add bounds checking for target_max_token to prevent index out of bounds
            vocab_size = len(self.model.t2d)
            if self.rank == 0 and torch.any(target_max_token >= vocab_size):
                logger.warning(
                    f"Some target tokens ({torch.max(target_max_token).item()}) exceed t2d vocab size ({vocab_size})"
                )

            # Clamp target_max_token to valid range
            target_max_token = torch.clamp(target_max_token, 0, vocab_size - 1)
            target_mask = self.model.t2d[target_max_token]
            target_mask = target_mask[..., None].int()
            position_mask = target_mask * loss_mask[..., None]

            target_logits_mapped = torch.index_select(target_logits, dim=-1, index=self.model.valid_t2d)
            target_logits_mapped = target_logits_mapped.float()
            target_p = nn.Softmax(dim=2)(target_logits_mapped).detach()

        # Prepare attention mask
        causal_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), hidden_states, 0)

        # Initialize iterative prediction
        plosses = []
        acces = []
        current_hidden = hidden_states.to(dtype=self.model.config.torch_dtype)
        current_input_ids = input_ids
        current_target_p = target_p
        current_position_mask = position_mask
        current_loss_mask = loss_mask

        # Eagle3 iterative prediction loop
        for step in range(self.prediction_length):
            # Get input embeddings
            inputs_embeds = self.model.model.embed_tokens(current_input_ids)
            inputs_embeds = inputs_embeds.to(dtype=self.model.config.torch_dtype)

            # Forward pass through Eagle3 model
            outputs = self.model_engine(
                base_model_hidden_states=current_hidden,
                input_ids=current_input_ids,
                attention_mask=causal_mask,
                output_hidden_states=True,
            )

            # Get output hidden states and compute logits
            predict_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[0]
            predict_hidden = self.model.model.norm(predict_hidden)
            logits = self.model.lm_head(predict_hidden)
            logits = logits.float()

            # Compute loss for this step
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = current_target_p * out_logp
            step_loss = -torch.sum(current_position_mask * plogp, 2).mean()
            plosses.append(step_loss)

            # Compute accuracy for this step
            with torch.no_grad():
                correct = ((logits.argmax(-1) == current_target_p.argmax(-1)) * current_position_mask.squeeze(-1)).sum().item()
                total = current_position_mask.sum().item() + 1e-6
                acces.append(correct / total)

            # Prepare for next iteration (shift everything)
            if step < self.prediction_length - 1:
                current_input_ids = self._padding(current_input_ids, left=False)
                current_target_p = self._padding(current_target_p, left=False)
                current_position_mask = self._padding(current_position_mask, left=False)
                current_loss_mask = self._padding(current_loss_mask, left=False)
                current_hidden = predict_hidden  # Use predicted hidden states for next step

                # Update attention mask for next step
                step_indices = torch.arange(seq_length, device=attention_mask.device)
                step_indices_from = step_indices[step:]
                step_indices_to = step_indices[: seq_length - step]
                if len(step_indices_from) > 0 and len(step_indices_to) > 0:
                    causal_mask[:, :, step_indices_from, step_indices_to] = torch.finfo(causal_mask.dtype).min

        return plosses, acces

    def train(self):
        if self.rank == 0:
            tracking = Tracking(
                project_name=self.args.project_name, experiment_name=self.args.experiment_name, default_backend="wandb"
            )
        else:
            tracking = None

        for epoch in range(self.start_epoch, self.args.epochs):
            self.model_engine.train()

            if self.rank == 0:
                train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                train_iter = self.train_loader

            # Track epoch metrics for each prediction step
            epoch_plosses = [[] for _ in range(self.prediction_length)]
            epoch_acces = [[] for _ in range(self.prediction_length)]

            for batch_idx, batch in enumerate(train_iter):
                batch = {k: v.cuda() for k, v in batch.items()}
                batch["hidden_states"] = batch["hidden_states"].to(dtype=self.model.config.torch_dtype)

                plosses, acces = self._compute_loss(batch)

                ploss_weights = [0.8**i for i in range(len(plosses))]
                total_loss = sum([ploss_weights[i] * plosses[i] for i in range(len(plosses))])

                self.model_engine.backward(total_loss)
                self.model_engine.step()

                # Store metrics for each prediction step
                for i in range(len(plosses)):
                    epoch_plosses[i].append(plosses[i].item())
                    epoch_acces[i].append(acces[i])

                if self.rank == 0:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    metrics = {
                        "train/total_loss": total_loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }

                    # Log metrics for each prediction step
                    for i in range(len(plosses)):
                        metrics[f"train/ploss_{i}"] = plosses[i].item()
                        metrics[f"train/acc_{i}"] = acces[i]

                    tracking.log(metrics, step=global_step)

                    # Show average accuracy in progress bar
                    avg_acc = sum(acces) / len(acces)
                    train_iter.set_postfix({"loss": f"{total_loss.item():.4f}", "avg_acc": f"{avg_acc:.2%}", "epoch": epoch})

            # Log epoch averages
            if self.rank == 0:
                for i in range(self.prediction_length):
                    avg_ploss = sum(epoch_plosses[i]) / len(epoch_plosses[i])
                    avg_acc = sum(epoch_acces[i]) / len(epoch_acces[i])

                    logger.info(
                        f"Train Epoch [{epoch + 1}/{self.args.epochs}], Step {i}, pLoss: {avg_ploss:.4f}, Acc: {avg_acc:.2%}"
                    )

                    tracking.log(
                        {
                            f"train/epoch_ploss_{i}": avg_ploss,
                            f"train/epoch_acc_{i}": avg_acc,
                        },
                        step=epoch,
                    )

            client_state = {
                "epoch": epoch,
                "step": epoch * len(self.train_loader) + len(self.train_loader) - 1,
            }
            self.model_engine.save_checkpoint(self.args.output_dir, client_state=client_state, exclude_frozen_parameters=False)

            if self.rank == 0:
                logger.info(f"Checkpoint saved at epoch {epoch}: {self.args.output_dir}")


def main():
    parser = add_args()
    args = parser.parse_args()

    # Initialize distributed training
    deepspeed.init_distributed()

    # Create trainer and start training
    trainer = Eagle3TrainerDeepSpeed(args)
    trainer.train()


if __name__ == "__main__":
    main()
