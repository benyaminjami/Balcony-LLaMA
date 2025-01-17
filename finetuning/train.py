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
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers_extra import *
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

from train_config import SFTDistillConfig

import torch.distributed as dist
from datetime import timedelta
# from aim.hugging_face import AimCallback

logger = logging.getLogger(__name__)

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

    ###############
    # Load datasets
    ###############
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
    # raw_datasets = load_from_disk("/ephemeral/parsa/datasets")

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    
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
        tie_exit_lm_head=True,
        output_exit_layers=[4],
        output_full_model=True,
        exit_layer_indices=[4, 8, 12],
        exit_decoder_layer=True,
    )
    
    if training_args.unfreeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in training_args.unfreeze_layers):
                param.requires_grad = False
    
    # model.config.num_hidden_layers = 4

    # # config

    pt_config = PromptTuningConfig(
        peft_type="PROMPT_TUNING",
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        token_dim=model.config.hidden_size,
        num_transformer_submodules=1,
        num_attention_heads=model.config.num_attention_heads,
        num_layers=model.config.num_hidden_layers,
    )

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
        model=model,
        train_dataset=train_dataset.to_iterable_dataset(),
        # eval_dataset=eval_dataset.to_iterable_dataset(),
        args=training_args,
        peft_config=pt_config,
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