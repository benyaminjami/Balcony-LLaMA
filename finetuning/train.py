##!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys
import os

import datasets
from datasets import load_from_disk

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from transformers_extra.models.nested_llama_wrapper.modeling_nested_llama_wrapper import LlamaEarlyExitWrapper, LlamaConfigWrapper
from peft import PromptEmbedding, PromptTuningConfig, MultitaskPromptTuningConfig, TaskType, PrefixTuningConfig

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import setup_chat_format, SFTTrainer
from kd_trainer import KDTrainer
from metatoken_learning import metatoken_save_pretrained

from peft import get_peft_model, LoraConfig

from train_config import SFTDistillConfig

import torch.distributed as dist
from datetime import timedelta
# from aim.hugging_face import AimCallback
# from transformers.data.data_collator import DataCollatorWithPadding, pad_without_fast_tokenizer_warning
# from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# class MetaDataCollator(DataCollatorWithPadding):
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         batch = pad_without_fast_tokenizer_warning(
#             self.tokenizer,
#             features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=self.return_tensors,
#         )
#         if "label" in batch:
#             batch["labels"] = batch["label"]
#             del batch["label"]
#         if "label_ids" in batch:
#             batch["labels"] = batch["label_ids"]
#             del batch["label_ids"]
#         return batch
    
def main():
    
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=360000))

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    tokenizer = get_tokenizer(model_args, data_args)

    ###############
    # Load datasets
    ###############
    if training_args.load_from_disk is None or not os.path.exists(training_args.load_from_disk):
        raw_datasets = get_datasets(
            data_args,
            splits=data_args.dataset_splits,
            configs=data_args.dataset_configs,
            # columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
            columns_to_keep=["prompt", "text"],
        )

        # Transformation function
        def add_messages_column(example):
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["text"]}
            ]
            return {"messages": messages}

        raw_datasets = raw_datasets.map(add_messages_column, num_proc=data_args.preprocessing_num_workers, remove_columns=["prompt", "text"])

        logger.info(
            f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
        )
        column_names = list(raw_datasets["train"].features)
        
        ################
        # Load tokenizer
        ################
        
        model = model_args.model_name_or_path
        # For ChatML we need to add special tokens and resize the embedding layer
        # if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        #     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        #     model, tokenizer = setup_chat_format(model, tokenizer)
        #     model_kwargs = None

        #####################
        # Apply chat template
        #####################
        raw_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )
        ##########################
        # Decontaminate benchmarks
        ##########################
        if training_args.decontaminate:
            num_raw_train_samples = len(raw_datasets["train"])
            raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=data_args.preprocessing_num_workers)
            num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
            logger.info(
                f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
            )
        raw_datasets.save_to_disk(training_args.load_from_disk, num_shards={'train':64}, num_proc=64)
    else:
        raw_datasets = datasets.load_from_disk(training_args.load_from_disk)

    # raw_datasets.save_to_disk("/ephemeral/parsa/datasets")

    train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        tie_exit_lm_head=training_args.tie_exit_lm_head,
        output_exit_layers=training_args.output_exit_layers,
        output_full_model=training_args.output_full_model,
        exit_layer_indices=training_args.exit_layer_indices,
        exit_decoder_layer=training_args.exit_decoder_layer,
    )
    
    # Initilize balcony weights
    if not training_args.random_balcony_initialization:
        for i, exit_module in enumerate(model.model.exit_modules):
            layer_idx = model.config.exit_layer_indices[i]
            if model.config.exit_decoder_layer or model.config.exit_mlp:
                exit_module[0].load_state_dict(model.model.layers[-1].state_dict())
                exit_module[1].load_state_dict(
                    model.model.layers[layer_idx].input_layernorm.state_dict()
                )
                # print('FFFFFFFF', layer_idx, )
            else:
                exit_module[0].load_state_dict(
                    model.model.layers[layer_idx].input_layernorm.state_dict()
                )
    
    
    if training_args.freeze_model:
        for name, param in model.named_parameters():
            if any(layer in name for layer in training_args.unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # config

    if training_args.meta_training:
        assert training_args.adapter_type in ['prompt', 'prefix', 'lora'], "adapter_type should be in ['prompt', 'prefix', 'lora']!!!"
        PeftModel.save_pretrained = metatoken_save_pretrained

        if training_args.adapter_type == 'prompt':
            peft_config = PromptTuningConfig(
                peft_type="PROMPT_TUNING",
                task_type="CAUSAL_LM",
                num_virtual_tokens=100,
                token_dim=model.config.hidden_size,
                num_transformer_submodules=1,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=model.config.num_hidden_layers,
            )
        elif training_args.adapter_type == 'prefix':
            peft_config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING",
                task_type="CAUSAL_LM",
                num_virtual_tokens=1,
                token_dim=model.config.hidden_size,
                num_transformer_submodules=1,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=training_args.output_exit_layers[0],
            )
        elif training_args.adapter_type == 'lora':
            peft_config = LoraConfig(
                peft_type="LoRA_TUNING",
                layers_to_transform=None,
                layers_pattern=None,
                task_type="CAUSAL_LM",  # Type of task (e.g., CAUSAL_LM, SEQ_CLS, TOKEN_CLS)
                inference_mode=False,          # Set to True if you're using the model for inference
                bias="none",
                r=8,                           # Rank of the LoRA adaptation matrix
                lora_alpha=32,                 # Scaling factor for LoRA
                lora_dropout=0.1,              # Dropout rate for LoRA
                target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA (example for LLaMA)
            )
            
        peft_model = get_peft_model(model, peft_config)

        if training_args.adapter_type == 'lora':
            teacher_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                output_exit_layers=[],
                output_full_model=True,
                exit_layer_indices=[],
            )
        
        # Unfreeze parameters containing "exit_modules" in their name
        for name, param in peft_model.named_parameters():
            if "exit_modules" in name:
                param.requires_grad = True
                print(f"Unfroze parameter: {name}")


    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # pre_config = PrefixTuningConfig(
    #     peft_type="PREFIX_TUNING",
    #     task_type="CAUSAL_LM",
    #     num_virtual_tokens=20,
    #     token_dim=model.config.hidden_size,
    #     num_transformer_submodules=1,
    #     num_attention_heads=model.config.num_attention_heads,
    #     num_layers=model.config.num_hidden_layers,
    # )
    # mpt_config = MultitaskPromptTuningConfig(
    #     num_tasks=3,
    #     task_type=TaskType.CAUSAL_LM,
    #     num_virtual_tokens=20,
    #     num_transformer_submodules=1,
    # )

    # aim_callback = AimCallback(experiment=training_args.output_dir.split('/')[-1])

    # setup the trainer
    trainer = KDTrainer(
        model=peft_model if training_args.meta_training else model,
        train_dataset=train_dataset.to_iterable_dataset(),
        teacher_model=teacher_model if training_args.meta_training and training_args.adapter_type == 'lora' else None,
        # eval_dataset=eval_dataset.to_iterable_dataset(),
        args=training_args,
        # peft_config=pt_config,
        # data_collator=MetaDataCollator(),
        tokenizer=tokenizer,
        # callbacks=[aim_callback],
        # dataset_kwargs=training_args.dataset_kwargs,
    )

    # launch
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(model_args.model_name_or_path, "final_checkpoint/"))

    # if args.save_merged_model:
    #     # Free memory for merging weights
    #     del model
    #     if is_torch_xpu_available():
    #         torch.xpu.empty_cache()
    #     elif is_torch_npu_available():
    #         torch.npu.empty_cache()
    #     else:
    #         torch.cuda.empty_cache()

    #     model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    #     model = model.merge_and_unload()

    #     output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
    #     model.save_pretrained(output_merged_dir, safe_serialization=True)

    #     if args.push_to_hub:
    #         model.push_to_hub(args.repo_id, "Upload model")
    
    print("Training Done! 💥")


if __name__ == "__main__":
    main()