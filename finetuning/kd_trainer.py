from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from transformers import PreTrainedModel, is_wandb_available

from trl import SFTTrainer
from hf_mtask_trainer import HfMultiTaskTrainer

SFTTrainer.__bases__ = (HfMultiTaskTrainer,)

from train_config import SFTDistillConfig

class KDTrainer(SFTTrainer):
    _tag_names = ["trl", "kd"]

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[SFTDistillConfig] = None,
        *sft_args,
        **kwargs,
    ):

        super().__init__(*sft_args, args=args, **kwargs)

        self.kl_weight = args.kl_weight
        self.ce_weight = args.ce_weight
        self.teacher_model = None
        if teacher_model is not None:
            self.teacher_model = (
                self.accelerator.prepare(teacher_model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute teacher output in eval mode
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            return_dict=True
        )

        logits = outputs.logits
        model_config = model.module.model.config
        
        total_ce_loss = sum(outputs.losses) / len(outputs.losses)
        
        if model.module.model.config.output_full_model:
            teacher_logits = logits[-1].detach()
            logits = logits[:-1]
            
            self.additional_state.add_metrics(**{f'ce_loss_{model_config.num_hidden_layers}': outputs.losses[-1]})
            outputs.losses = outputs.losses[:-1]

        if self.args.meta_training:
            assert not model.module.config.output_full_model, "Meta Token Training Requires output_full_model = False!"
            assert model.module.config.output_exit_layers is not None, "Meta Token Training Requires output_exit_layers not None!"

            # Adapt model configs
            model.module.config.output_full_model = True
            output_exit_layers_tmp = model.module.config.output_exit_layers
            model.module.config.output_exit_layers = None #[model.module.config.num_hidden_layers]
            model.module.base_model.model.num_forward_layers = model.module.config.num_hidden_layers
            
            with torch.no_grad():
                if self.teacher_model is None:
                    with model.module.disable_adapter():
                        teacher_outputs = model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            return_dict=True
                        )
                else:
                    teacher_outputs = self.teacher_model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            return_dict=True
                        )
                # model.module.enable_adapters()
            
            # Reverse model configs
            model.module.config.output_full_model = False
            model.module.config.output_exit_layers = output_exit_layers_tmp
            model.module.base_model.model.num_forward_layers = max(output_exit_layers_tmp)

            teacher_logits = teacher_outputs.logits
            teacher_logits = teacher_logits[-1].detach()
            self.additional_state.add_metrics(**{f'ce_loss_{model_config.num_hidden_layers}': teacher_outputs.losses[-1]})
        
        for i, loss in enumerate(outputs.losses):
            self.additional_state.add_metrics(**{f'ce_loss_{model_config.output_exit_layers[i]}': loss})
            
        total_kl_loss = 0
        if self.kl_weight > 0:
            # assert model.module.model.config.output_full_model, "KL loss requires full model output"
            kl_losses = []
            for i, lg in enumerate(logits):
                if self.args.meta_training and self.args.adapter_type == 'prompt':
                    num_virtual_tokens = model.module.peft_config['default'].num_virtual_tokens
                    lg = lg[:, num_virtual_tokens:, :]
                kl_loss = F.kl_div(F.log_softmax(lg, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
                kl_losses.append(kl_loss)
                self.additional_state.add_metrics(**{f'kl_loss_{model_config.output_exit_layers[i]}': kl_loss})

            total_kl_loss = sum(kl_losses) / len(kl_losses)
        # compute cross entropy loss
        loss = self.kl_weight * total_kl_loss + self.ce_weight * total_ce_loss
        # Return loss
        return (loss, outputs) if return_outputs else loss
