import os
import json
import torch
import torch.nn as nn

from fastchat.utils import str_to_torch_dtype
from transformers.models.llama import LlamaConfig
from transformers import AutoModelForCausalLM
from collections import OrderedDict

try:
    from transformers_extra import *
except:
    pass

from kangaroo.adapter import AdapterModel
from kangaroo.earlyexit import EarlyExitLlamaForCausalLM

class KangarooModel(nn.Module):

    def __init__(
            self,
            balcony=True,
            raw=False,
            **kwargs
    ):
        super().__init__()
        if balcony:
            if raw:
                self.init_balcony_raw(**kwargs)
            else:
                self.init_balcony(**kwargs)
        else:
            self.init_kangaroo(**kwargs)
    
    def init_balcony_raw(
        self,
        base_model_name_or_path,
        args,
        EARLY_STOP_LAYER
    ):
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto", EARLY_STOP_LAYER = EARLY_STOP_LAYER)
        self.base_model = self.base_model.eval()
        
        config = LlamaConfig.from_pretrained(os.path.join(base_model_name_or_path, "config.json"), num_hidden_layers=1)

        model = AutoModelForCausalLM.from_pretrained("/work/benyamin/smollm/hf_checkpoints/Smollm_1B_Frozen_Sorted_Balcony")
        
        final_state = OrderedDict()
        idx = model.config.exit_layer_indices.index(EARLY_STOP_LAYER)
        states = model.model.exit_modules[idx].state_dict()
        for k, v in states.items():
            if k[0] == '0':
                final_state['layers.' + k] = v
            if k[0] == '1':
                final_state['norm.weight'] = v
        # print(final_state)
        config = LlamaConfig.from_pretrained(os.path.join(base_model_name_or_path, "config.json"), num_hidden_layers=1)

        self.adapter_model = AdapterModel(config, balcony=True)
        self.adapter_model.load_state_dict(final_state, strict=True)
        self.adapter_model = self.adapter_model.eval().to(self.base_model.device)

        
        if args.dtype == "float16":
            self.adapter_model = self.adapter_model.half()

        self.head_model = self.base_model.lm_head

        if args.dtype == "float16":
            self.head_model = self.head_model.half()

    def init_balcony(
            self,
            base_model_name_or_path,
            adapter_model_path,
            args,
            EARLY_STOP_LAYER = 2
        ):
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto", EARLY_STOP_LAYER = EARLY_STOP_LAYER)
        self.base_model = self.base_model.eval()

        config = LlamaConfig.from_pretrained(os.path.join(base_model_name_or_path, "config.json"), num_hidden_layers=1)
        self.adapter_model = AdapterModel(config, balcony=True)

        self.adapter_model.load_state_dict(torch.load(os.path.join(adapter_model_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
        self.adapter_model = self.adapter_model.eval().to(self.base_model.device)

        if args.dtype == "float16":
            self.adapter_model = self.adapter_model.half()

        self.head_model = self.base_model.lm_head

        if args.dtype == "float16":
            self.head_model = self.head_model.half()


    
    def init_kangaroo(            
            self,
            base_model_name_or_path,
            adapter_model_path,
            args,
            EARLY_STOP_LAYER = 2
        ):
        self.base_model = EarlyExitLlamaForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=str_to_torch_dtype(args.dtype), device_map="auto", EARLY_STOP_LAYER = EARLY_STOP_LAYER)
        self.base_model = self.base_model.eval()

        config = LlamaConfig.from_pretrained(os.path.join(adapter_model_path, "config.json"))
        self.adapter_model = AdapterModel(config, balcony=False)

        self.adapter_model.load_state_dict(torch.load(os.path.join(adapter_model_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
        self.adapter_model = self.adapter_model.eval().to(self.base_model.device)
        self.adapter_model.to(torch.bfloat16)
        if args.dtype == "float16":
            self.adapter_model = self.adapter_model.half()

        self.head_model = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        with open(os.path.join(base_model_name_or_path, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(base_model_name_or_path, head_path), map_location='cpu')
        tensor = weights["lm_head.weight"].float()
        self.head_model.weight.data = tensor
        self.head_model = self.head_model.eval().to(self.base_model.device)

        if args.dtype == "float16":
            self.head_model = self.head_model.half()

    def forward(self):
        raise NotImplementedError






