from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_nestedllama import NestedLlamaConfig
from .modeling_nestedllama import NestedLlamaForCausalLM

AutoConfig.register("nested_llama", NestedLlamaConfig)
AutoModelForCausalLM.register(NestedLlamaConfig, NestedLlamaForCausalLM)
