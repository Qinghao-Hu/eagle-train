import os
import json
import logging
import torch
import torch.distributed as dist
import hydra
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "INFO"))


class EagleDatasetGenerator:

    def __init__(self, datapath, max_len=32768, base_model_path=None, save_dir=None, process_rank=0):
        self.max_len = max_len
        self.base_model_path = base_model_path
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_type = None
        self.processed_dataset = []
        self.process_rank = process_rank
        # Load tokenizer
        if base_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

            # Handle padding token
            if not self.tokenizer.pad_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

            # Detect model type for later use with separators
            if "deepseek" in base_model_path.lower():
                self.model_type = "deepseek"
                logger.info(f"Detected DeepSeek model type")
                self.sep_assistant = "<｜Assistant｜>\n\n"
                self.sep_user = "<｜User｜>"
            elif "qwen" in base_model_path.lower():
                self.model_type = "qwen"
                logger.info(f"Detected Qwen model type")
                self.sep_assistant = "<|im_end|>\n<|im_start|>assistant\n"
                self.sep_user = "<|im_end|>\n<|im_start|>user\n"
            elif "llama" in base_model_path.lower():
                self.model_type = "llama"
                logger.info(f"Detected Llama model type")
                # Llama3-style chat template
                self.sep_assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                self.sep_user = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            else:
                logger.info(f"Using generic model handling (no specific format detected)")
                # Default separators if needed
                self.sep_assistant = "<|im_end|>\n<|im_start|>assistant\n"
                self.sep_user = "<|im_end|>\n<|im_start|>user\n"

            # Calculate separator lengths
            self.sep_len_assistant = len(self.tokenizer(self.sep_assistant).input_ids)
            self.sep_len_user = len(self.tokenizer(self.sep_user).input_ids)
            logger.info(f"Separator lengths - Assistant: {self.sep_len_assistant}, User: {self.sep_len_user}")

        # Load the data using datasets library
        self.load_data(datapath)

        # # Distribute data to different workers
        # self.distribute_data()

        # # Process the dataset
        self.process_dataset(self.worker_data)

        # Load base model for getting hidden states
        if base_model_path:
            # Initialize base model
            logger.info(f"Rank {self.rank}/{self.world_size}: Loading base model...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,  # torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            self.base_model.eval()

            # Get model's maximum context length
            self.model_max_length = getattr(self.base_model.config, "max_position_embeddings", self.max_len)
            logger.info(f"Model's maximum context length: {self.model_max_length}")
            # Update max_len to be within model's limits
            self.max_len = 4096  # min(self.max_len, self.model_max_length)
            logger.info(f"Using maximum sequence length: {self.max_len}")

            self.worker_data = self.processed_dataset

    def load_data(self, datapath):
        """Load data using datasets library and process conversations"""
        # Try to load the dataset with appropriate format detection
        logger.info(f"Loading dataset from {datapath}")
        full_dataset = load_dataset("parquet", data_dir=datapath)["train"]
        # full_dataset = load_dataset("parquet", data_files=f"{datapath}/data/train-00000-of-00038.parquet")["train"]

        # Sample data according to process_rank
        # Each rank gets a different 5% slice of the data
        ratio = 0.05  # For OpenThoughts2-1M
        # ratio = 0.5  # For Eurus-320
        num_samples = len(full_dataset)
        slice_size = int(num_samples * ratio)  # 5% slice size

        # Calculate start and end indices for this rank's slice
        start_idx = self.process_rank * slice_size
        end_idx = min((self.process_rank + 1) * slice_size, num_samples)

        # Create indices for this rank's slice
        indices = list(range(start_idx, end_idx))

        # Select the data for this rank
        self.worker_data = full_dataset.select(indices)
        self.worker_indices = indices

        logger.info(f"Loaded {len(self.worker_data)} samples ({slice_size}/{num_samples}, {ratio*100}% of total data)")
        del full_dataset

    def process_dataset(self, ds):
        """Process items from a HuggingFace dataset or list of items"""
        if hasattr(ds, "column_names"):
            # Handle HuggingFace dataset
            if "conversations" in ds.column_names:
                # Process conversations field
                for item in ds:
                    self.process_conversation_item(item)
            elif "messages" in ds.column_names:
                # Process messages field
                for item in ds:
                    if isinstance(item["messages"], list):
                        messages = []
                        for msg in item["messages"]:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                messages.append({"role": msg["role"], "content": msg["content"]})

                        if len(messages) > 1:
                            self.create_conversation_entry(messages)
            elif all(field in ds.column_names for field in ["prompt", "response"]):
                # Handle simple prompt/response pairs
                for item in ds:
                    self.format_conversation(item["prompt"], item["response"])
        else:
            # Handle list of items
            for item in ds:
                if isinstance(item, dict):
                    if "conversations" in item:
                        self.process_conversation_item(item)
                    elif "messages" in item:
                        if isinstance(item["messages"], list):
                            messages = []
                            for msg in item["messages"]:
                                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                    messages.append({"role": msg["role"], "content": msg["content"]})
                            if len(messages) > 1:
                                self.create_conversation_entry(messages)
                    elif all(field in item for field in ["prompt", "response"]):
                        self.format_conversation(item["prompt"], item["response"])
                    else:
                        # Try to infer the format from the item
                        for key, value in item.items():
                            if isinstance(value, list) and len(value) >= 2:
                                # Might be a conversation
                                self.process_conversation_item({key: value})

    def process_conversation_item(self, item):
        """Process a single conversation item"""
        if hasattr(item, "conversations") or "conversations" in item:
            conversations = item.conversations if hasattr(item, "conversations") else item["conversations"]

            if len(conversations) >= 2:
                # messages = [{"role": "system", "content": "You are a helpful assistant."}]
                messages = []

                # Handle different conversation formats
                if isinstance(conversations[0], dict) and "value" in conversations[0]:
                    # ShareGPT format
                    roles = {"human": "user", "gpt": "assistant"}
                    source = conversations

                    # # Skip if first message is not from human
                    # if "from" in source[0] and roles.get(source[0]["from"]) != "user":
                    #     source = source[1:]

                    for j, sentence in enumerate(source):
                        # if "from" in sentence and sentence["from"] in roles:
                        role = sentence["from"]
                        content = sentence["value"]

                        # For Llama models, add a space before assistant responses
                        if self.model_type == "llama" and role == "assistant":
                            content = " " + content

                        messages.append({"role": role, "content": content})

                else:
                    # Simple alternating format
                    for i, content in enumerate(conversations):
                        role = "user" if i % 2 == 0 else "assistant"

                        text_content = ""
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, dict) and "value" in content:
                            text_content = content["value"]

                        # For Llama models, add a space before assistant responses
                        if self.model_type == "llama" and role == "assistant":
                            text_content = " " + text_content

                        messages.append({"role": role, "content": text_content})

                # Only process if we have valid messages
                if len(messages) > 1:
                    self.create_conversation_entry(messages)

    def create_conversation_entry(self, messages):
        """Create conversation entry from messages"""
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            logger.warning("Tokenizer not available for processing")
            return None

        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            # return_dict=True,
            # return_tensors="pt",
            # add_generation_prompt=True,
            # return_assistant_tokens_mask=True,
        )

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        input_ids = self.tokenizer(
            conversation,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        ).input_ids[0]

        loss_mask = torch.ones_like(input_ids)

        # Unified implementation for handling different model types
        if self.model_type in ["qwen", "llama", "deepseek"]:
            turns = conversation.split(self.sep_user)

            if len(turns) > 1:
                # First turn contains the system message, combine it with the first user message
                turns[1] = turns[0] + self.sep_user + turns[1]
                turns = turns[1:]  # Remove the system message turn

                cur_len = 1
                loss_mask[:cur_len] = 0  # Mask out the beginning token

                for i, turn in enumerate(turns):
                    if turn == "":
                        break

                    turn_len = len(self.tokenizer(turn).input_ids)

                    parts = turn.split(self.sep_assistant)
                    if len(parts) != 2:
                        break

                    parts[0] += self.sep_assistant
                    instruction_len = len(self.tokenizer(parts[0]).input_ids)

                    # Adjust the loss mask based on model type
                    if self.model_type == "qwen" or self.model_type == "deepseek":
                        if i == 0:
                            # For the first turn, mask out the user part
                            loss_mask[0 : cur_len + instruction_len - 2] = 0
                        else:
                            # For subsequent turns, mask out the user part
                            loss_mask[cur_len - 2 : cur_len + instruction_len - 1] = 0

                        cur_len += turn_len
                        cur_len += 2  # Adjust for separator
                    elif self.model_type == "llama":
                        if i == 0:
                            loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0

                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3

                # Mask out the rest of the sequence
                loss_mask[cur_len:] = 0

                # print(loss_mask)
                # print(self.sep_len_assistant, self.sep_len_user)  # Llama3: 6, 5; DS: 3, 2; Qwen: 5, 5
                # if self.rank == 0:
                #     print(conversation)
                #     print(self.tokenizer.decode(input_ids[loss_mask.bool()]))
                # exit()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.processed_dataset.append({"input_ids": input_ids, "loss_mask": loss_mask, "conversation": conversation})

        return True

    def format_conversation(self, prompt, response):
        """Format conversation from prompt/response pair"""
        if not prompt or not response:
            return None

        messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        return self.create_conversation_entry(messages)

    def distribute_data(self):
        """Distribute data to different workers"""
        num_samples = len(self.dataset)
        worker_indices = []

        for i in range(self.rank, num_samples, self.world_size):
            worker_indices.append(i)

        self.worker_indices = worker_indices
        self.worker_data = [self.dataset[i] for i in worker_indices]

        logger.info(f"Rank {self.rank}/{self.world_size} has {len(self.worker_indices)} samples")
        del self.dataset

    @torch.no_grad()
    def process_data(self):
        """Process data and generate hidden states with sequence length bucketing"""
        # Create bucket directories
        bucket_ranges = [
            (0, 2048, "0K-2K"),
            (2048, 4096, "2K-4K"),
            (4096, 8192, "4K-8K"),
            (8192, 16384, "8K-16K"),
            (16384, 32768, "16K-32K"),
        ]

        bucket_stats = {name: 0 for _, _, name in bucket_ranges}
        bucket_dirs = {}

        for _, _, name in bucket_ranges:
            bucket_dir = os.path.join(self.save_dir, name)
            os.makedirs(bucket_dir, exist_ok=True)
            bucket_dirs[name] = bucket_dir

        dropped_count = 0
        max_seq_len = 0

        # Create progress bar for this rank
        desc = f"Rank {self.rank}/{self.world_size}"
        pbar = tqdm(enumerate(self.worker_data), total=len(self.worker_data), desc=desc, position=self.rank)

        for i, data_point in pbar:
            orig_idx = self.worker_indices[i]
            input_ids = data_point["input_ids"]
            seq_len = len(input_ids)

            # # Skip sequences longer than 32K
            # if seq_len > 32768:
            if seq_len > 2048:
                dropped_count += 1
                continue

            # Find appropriate bucket
            bucket_name = None
            for start, end, name in bucket_ranges:
                if start <= seq_len < end:
                    bucket_name = name
                    bucket_stats[name] += 1
                    break

            if bucket_name is None:
                dropped_count += 1
                continue

            # Process the sequence
            input_ids = input_ids.unsqueeze(0).to(self.get_device())
            loss_mask = data_point["loss_mask"]

            max_seq_len = max(max_seq_len, seq_len)

            # Extract hidden states
            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1]

            # Store processed data
            processed_data = {
                "index": orig_idx,
                "input_ids": input_ids.cpu().squeeze(0),
                "hidden_state": hidden_state.cpu().squeeze(0),
                "loss_mask": loss_mask,
                "seq_len": seq_len,
            }

            # Save to appropriate bucket directory
            output_path = os.path.join(bucket_dirs[bucket_name], f"data_{orig_idx}.pt")
            torch.save(processed_data, output_path)

            # Update progress bar postfix with current bucket info
            pbar.set_postfix(bucket=bucket_name, seq_len=seq_len)

        # # Gather max sequence length across all ranks
        # if self.world_size > 1:
        #     max_seq_len_tensor = torch.tensor(max_seq_len, device=self.get_device())
        #     dist.all_reduce(max_seq_len_tensor, op=dist.ReduceOp.MAX)
        #     max_seq_len = max_seq_len_tensor.item()

        # Save statistics (only rank 0)
        if self.rank == 0:
            stats = {"max_seq_len": max_seq_len, "bucket_stats": bucket_stats, "dropped_count": dropped_count}
            stats_file = os.path.join(self.save_dir, "bucket_stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved bucket statistics to {stats_file}")
            logger.info(f"Bucket distribution: {bucket_stats}")
            logger.info(f"Dropped sequences: {dropped_count}")

    def get_device(self):
        """Get the device for the current process"""
        if hasattr(self.base_model, "device"):
            return self.base_model.device
        else:
            return f"cuda:{self.local_rank}"


@hydra.main(config_path="config", config_name="datagen_config", version_base=None)
def main(config):
    generator = EagleDatasetGenerator(
        datapath=config.data.data_path,
        max_len=config.data.max_length,
        base_model_path=config.model.base_model_path,
        save_dir=config.data.save_dir,
        process_rank=config.data.process_rank,
    )
    generator.process_data()


if __name__ == "__main__":
    main()  # Hydra automatically provides the config parameter

# Example command to run:
# srun -J eagle_datagen -N 1 --exclusive torchrun --standalone --nnodes=1 --nproc_per_node=8 -m fastrl.trainer.eagle_datagen_sft
