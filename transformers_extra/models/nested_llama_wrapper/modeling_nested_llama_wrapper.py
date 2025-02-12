import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, KwargsForCausalLM
from typing import Callable, List, Optional, Tuple, Union
from transformers.processing_utils import Unpack

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

class LlamaConfigWrapper(LlamaConfig):
    def __init__(
        self, 
        exit_layers=[],
        balcony=True,
        tie_exit_lm_head=None,
        total_num_layers=None,
        **kwargs):
        """
        Wrapper around the LlamaConfig to allow additional configuration handling.

        Args:
            kwargs: Additional arguments for configuration.
        """
        super().__init__(**kwargs)
        self.exit_layers = exit_layers
        self.balcony = balcony
        self.tie_exit_lm_head = tie_exit_lm_head if tie_exit_lm_head else self.tie_word_embeddings
        self.total_num_layers = total_num_layers if total_num_layers else self.num_hidden_layers
        self.num_hidden_layers = total_num_layers
        self.output_exit_layer = self.num_hidden_layers
    
    def add_exit_layers(self, exit_layers):
        if isinstance(exit_layers, int):
            self.exit_layers.append(exit_layers)
        else:
            self.exit_layers.extend(exit_layers)
    

class LlamaEarlyExitWrapper(LlamaForCausalLM):
    def __init__(self, config: LlamaConfigWrapper):
        """
        Initialize the LlamaEarlyExitWrapper.

        Args:
            base_model_name_or_path (str): Path or name of the base Llama model.
            exit_layers (list[int], optional): List of layer indices to add early exits. Defaults to None.
        """
        super(LlamaEarlyExitWrapper, self).__init__(config)

        exit_decoder_layers = {}
        exit_norms = {}
        exit_heads = {}
        for layer_idx in exit_decoder_layers:
            exit_decoder_layers[layer_idx] = LlamaDecoderLayer(config, config.num_hidden_layers+layer_idx)
            exit_norms[layer_idx] = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.tie_exit_lm_head:
                exit_heads[layer_idx] = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        exit_decoder_layers[config.num_hidden_layers] = self.model.layers[-1]
        exit_norms[config.num_hidden_layers] = self.model.norm
        exit_heads[config.num_hidden_layers] = self.lm_head
        
        self.exit_decoder_layers = nn.ModuleDict(exit_decoder_layers)
        self.exit_norms = nn.ModuleDict(exit_norms)
        self.exit_heads = nn.ModuleDict(exit_heads)
        
        self.exit_norm = self.model.norm
        self.exit_decoder_layer = self.model.layers[-1]
        self.model.norm = nn.Identity()
        
        config.num_hidden_layers = config.num_hidden_layers - 1
        
        self.output_exit_layer = None
        
        self.set_exit_layer(config.num_hidden_layers)
        
        self.post_init()

    def _add_exit_layer(self, layer_idx):
        """Add an early exit layer at the specified index."""
        self.config.exit_layers.append(layer_idx)
        self.exit_decoder_layers[layer_idx] = LlamaDecoderLayer(self.config, self.config.num_hidden_layers+layer_idx)
        self.exit_norms[layer_idx] = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        if self.config.tie_exit_lm_head:
            self.exit_heads[layer_idx] = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def add_exit(self, layer_idx):
        """Add an early exit at the specified layer index."""
        if layer_idx not in self.exit_modules:
            self._add_exit_layer(layer_idx)

    def set_exit_layer(self, exit_layer):
        """
        Set exit layers to the specified list of layers. Resets previous exit layers.

        Args:
            layers (list[int]): List of layer indices to set as exit layers.
        """
        self.output_exit_layer = exit_layer
        if not self.config.tie_exit_lm_head:
            self.lm_head = self.exit_heads[exit_layer]
        if self.config.balcony:
            self.exit_decoder_layer = self.exit_decoder_layers[exit_layer]

    def forward(
        self,
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
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

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

    def exit(
        self,
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
        num_logits_to_keep: int = 0,
        exit_layer: int = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.config.balcony:
            exit_layer = self.output_exit_layer

# Example Usage
# wrapper = LlamaEarlyExitWrapper("path/to/llama/model", exit_layers=[3, 6, 9])
# wrapper.add_exit(12)
# wrapper.set_exit_layers([2, 4, 8])
# outputs = wrapper(input_ids, attention_mask)
