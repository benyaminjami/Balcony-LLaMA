
from dataclasses import dataclass, field
from typing import Any, List, NewType
from alignment import SFTConfig

DataClassType = NewType("DataClassType", Any)

@dataclass
class SFTDistillConfig(SFTConfig):
    """
    Arguments related to the distillation process.
    """
    decontaminate: bool = field(default=False, metadata={"help": "Whether to apply the decontaminate steps."})
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio of KL loss."},
    )
    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss."},
    )
    freeze_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the model."},
    )
    unfreeze_layers: List[str] = field(
        default=None,
        metadata={"help": "Freeze layers."},
    )
    load_from_disk: str = field(
        default=None,
        metadata={"help": "load from disk flag."},
    )
    meta_training: bool = field(
        default=False,
        metadata={"help": "Training with metadata."},
    )
    tie_exit_lm_head: bool = field(
        default=False,
        metadata={"help": "Training with tied lm head."},
    )    
    output_full_model: bool = field(
        default=False,
        metadata={"help": "Getting the full model output."},
    )
    exit_decoder_layer: bool = field(
        default=False,
        metadata={"help": "Training with decoder layer in balcony modules."},
    )
    exit_mlp: bool = field(
        default=False,
        metadata={"help": "Training with mlp layer in balcony modules."},
    )
    exit_attention: bool = field(
        default=False,
        metadata={"help": "Training with attention module in balcony modules."},
    )
    adapter_type: str = field(
        default=None,
        metadata={"help": "Select the adapter among ['prompt', 'prefix', 'lora']"},
    )
    output_exit_layers: List[int] = field(
        default=None,
        metadata={"help": "output_exit_layers."},
    )
    exit_layer_indices: List[int] = field(
        default=None,
        metadata={"help": "exit_layer_indices."},
    )
    random_balcony_initialization: bool = field(
        default=False,
        metadata={"help": "Initialize balcony weights randomly."},
    )
    