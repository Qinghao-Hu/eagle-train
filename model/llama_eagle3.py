from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaModel as LlamaModelTF,
    LlamaRotaryEmbedding,
    LlamaMLP,
    LlamaAttention,
    LlamaRMSNorm,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, logging

logger = logging.get_logger(__name__)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # NOTE: Override the qkv projection for Eagle-3
        self.self_attn.q_proj = nn.Linear(
            config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.k_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.v_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: Add a hidden_norm for Eagle-3
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # NOTE: Concatenate the input_embeds and hidden_states for Eagle-3
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaModel(LlamaModelTF):
    def __init__(self, config: LlamaConfig):
        # super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: Add a midlayer, fc for Eagle-3
        self.midlayer = LlamaDecoderLayer(config, 0)
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(config.target_hidden_size * 3, config.hidden_size, bias=False)
        else:
            self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,  # NOTE: Modified here, can from the base model or self-generated
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        q_hidden_states: Optional[torch.Tensor] = None,  # Add q_hidden_states parameter
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # (Qinghao): Modified here
        inputs_embeds = inputs_embeds.to(base_model_hidden_states.dtype)
        hidden_states = base_model_hidden_states

        # Use q_hidden_states if provided
        if q_hidden_states is not None:
            # Get the last element of the sequence
            last_q_hidden_states = q_hidden_states[-1]
            hidden_states = last_q_hidden_states

        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                self.midlayer.__call__,
                inputs_embeds,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = self.midlayer(
                inputs_embeds,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        self.post_init()

        d2t = torch.zeros((config.draft_vocab_size), dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        q_hidden_states: Optional[torch.Tensor] = None,  # Add q_hidden_states parameter
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            base_model_hidden_states=base_model_hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            q_hidden_states=q_hidden_states,  # Pass q_hidden_states to model
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, torch_dtype, *model_args, **kwargs):
        # Create model with the provided config
        config = kwargs.get("config", None)
        if config is None:
            raise ValueError("Config must be provided when loading a model from pretrained.")

        # Initialize the model with the config
        model = cls(config, *model_args)

        # Try to load model weights
        try:
            import os
            from safetensors.torch import load_file as safe_load_file
            import torch
            from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

            # Check if it's a directory or a file path
            if os.path.isdir(pretrained_model_name_or_path):
                safe_path = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                pt_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)

                if os.path.exists(safe_path):
                    logger.info(f"Loading weights from safetensors file: {safe_path}")
                    state_dict = safe_load_file(safe_path)
                elif os.path.exists(pt_path):
                    logger.info(f"Loading weights from PyTorch file: {pt_path}")
                    state_dict = torch.load(pt_path, map_location="cpu")
                else:
                    raise ValueError(f"No model weights found in {pretrained_model_name_or_path}")
            else:
                # Assume it's a direct file path
                if pretrained_model_name_or_path.endswith(".safetensors"):
                    logger.info(f"Loading weights from safetensors file: {pretrained_model_name_or_path}")
                    state_dict = safe_load_file(pretrained_model_name_or_path)
                else:
                    logger.info(f"Loading weights from PyTorch file: {pretrained_model_name_or_path}")
                    state_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")

            # Load the state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if len(missing_keys) > 0:
                logger.warning(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            logger.info("Successfully loaded the weights into the model.")
        except Exception as e:
            logger.warning(f"Error loading state dict: {e}")
            logger.warning("Using model with random initialization.")

        return model.to(torch_dtype)

    def load_state_dict(self, state_dict, strict=True):
        # Fix the state_dict keys to match the expected structure
        new_state_dict = {}

        # Print original keys for debugging
        logger.info(f"Original state_dict keys: {list(state_dict.keys())}")

        # Handle model prefix for relevant keys
        for key, value in state_dict.items():
            if key in ["embed_tokens.weight", "norm.weight"]:
                new_state_dict[f"model.{key}"] = value
            elif key.startswith("midlayer."):
                new_state_dict[f"model.{key}"] = value
            elif key == "fc.weight":
                new_state_dict["model.fc.weight"] = value
            else:
                new_state_dict[key] = value

        # Print modified keys for debugging
        logger.info(f"Modified state_dict keys: {list(new_state_dict.keys())}")

        # Get model expected keys for comparison
        model_state = self.state_dict()
        logger.info(f"Model expected keys: {list(model_state.keys())}")

        # Now load with the fixed state dict
        return nn.Module.load_state_dict(self, new_state_dict, strict=False)
