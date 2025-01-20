from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftType,
    TaskType,
    _get_batch_size,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    id_tensor_storage,
    infer_device,
    load_peft_weights,
    map_cache_to_layer_device_map,
    set_peft_model_state_dict,
    shift_tokens_right,
)
from typing import Any, Literal, Optional, Union
import os
import warnings
import collections
import torch
import copy
from safetensors.torch import save_file as safe_save_file

def metatoken_save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters: Optional[list[str]] = None,
    save_embedding_layers: Union[str, bool] = "auto",
    is_main_process: bool = True,
    path_initial_model_for_weight_conversion: Optional[str] = None,
    **kwargs: Any,
) -> None:
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
    method.

    Args:
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        safe_serialization (`bool`, *optional*):
            Whether to save the adapter files in safetensors format, defaults to `True`.
        selected_adapters (`List[str]`,  *optional*):
            A list of adapters to be saved. If `None`, will default to all adapters.
        save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
            embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
            and automatically sets the boolean flag. This only works for 🤗 transformers models.
        is_main_process (`bool`, *optional*):
            Whether the process calling this is the main process or not. Will default to `True`. Will not save the
            checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
        path_initial_model_for_weight_conversion (`str, *optional*`):
            The path to the initialized adapter, which is obtained after initializing the model with PiSSA or OLoRA
            and before performing any training. When `path_initial_model_for_weight_conversion` is not None, the
            difference in adapter before and after fine-tuning is calculated. This difference can be represented as
            the parameters of a standard LoRA adapter. Using this converted adapter does not require changes to the
            base model, thus conveniently allowing the use of multiple PiSSA or OLoRA adapters with LoRA adapters,
            and the activation or deactivation of any adapters. Note that this conversion is not supported if
            `rslora` is used in combination with `rank_pattern` or `alpha_pattern`.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.

    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

    if selected_adapters is None:
        selected_adapters = list(self.peft_config.keys())
    else:
        if any(
            selected_adapter_name not in list(self.peft_config.keys())
            for selected_adapter_name in selected_adapters
        ):
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                f" {list(self.peft_config.keys())} - got {selected_adapters}."
            )

    def save_mutated_as_lora(peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs):
        if peft_config.use_rslora and (peft_config.rank_pattern or peft_config.alpha_pattern):
            msg = (
                "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not supported when "
                "using `rank_pattern` or `alpha_pattern` at the same time as `use_rslora=True`."
            )
            raise ValueError(msg)

        if not any(
            str(peft_config.init_lora_weights).lower().startswith(prefix) for prefix in ["pissa", "olora", "true"]
        ):
            warnings.warn(
                "`path_initial_model_for_weight_conversion` only works for converting a PiSSA or OLoRA adapter to "
                "a LoRA adapter"
            )
        initial_adapter_name = os.path.basename(path_initial_model_for_weight_conversion)
        try:
            self.load_adapter(
                os.path.dirname(path_initial_model_for_weight_conversion),
                subfolder=initial_adapter_name,
                adapter_name=initial_adapter_name,
            )
            is_pissa = str(self.peft_config[initial_adapter_name].init_lora_weights).lower().startswith("pissa")
            is_olora = str(self.peft_config[initial_adapter_name].init_lora_weights).lower() == "olora"
            if is_pissa or is_olora:
                raise ValueError(
                    "The `init_lora_weights` parameter of the initial adapter should be set to `True`. "
                    "Otherwise, `self.load_adapter` will subtract the decomposed values again based on the "
                    "residual model."
                )
            output_state_dict = self.base_model.subtract_mutated_init(
                output_state_dict, initial_adapter_name, kwargs
            )
        finally:
            self.delete_adapter(initial_adapter_name)
        return output_state_dict

    if is_main_process:
        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

    for adapter_name in selected_adapters:
        peft_config = self.peft_config[adapter_name]
        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
            save_embedding_layers=save_embedding_layers,
        )
        output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
        os.makedirs(output_dir, exist_ok=True)

        if is_main_process and safe_serialization:
            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in output_state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
            if path_initial_model_for_weight_conversion is not None:
                peft_config = copy.deepcopy(peft_config)
                peft_config.init_lora_weights = True
                peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                output_state_dict = save_mutated_as_lora(
                    peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                )
            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        elif is_main_process:
            if path_initial_model_for_weight_conversion is not None:
                peft_config = copy.deepcopy(peft_config)
                peft_config.init_lora_weights = True
                peft_config.save_pretrained(path_initial_model_for_weight_conversion)
                output_state_dict = save_mutated_as_lora(
                    peft_config, path_initial_model_for_weight_conversion, output_state_dict, kwargs
                )
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if peft_config.is_prompt_learning
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=peft_config.is_prompt_learning,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if is_main_process:
            if path_initial_model_for_weight_conversion is not None:
                peft_config.init_lora_weights = True
                peft_config.r *= 2
                if not peft_config.use_rslora:
                    peft_config.lora_alpha *= 2
                else:
                    # with rslora, we have scaling = alpha / sqrt(r), we thus adjust alpha to keep the same scaling
                    peft_config.lora_alpha *= 2**0.5

                if peft_config.rank_pattern:
                    peft_config.rank_pattern = {key: 2 * val for key, val in peft_config.rank_pattern.items()}
                if peft_config.alpha_pattern:
                    peft_config.alpha_pattern = {key: 2 * val for key, val in peft_config.alpha_pattern.items()}

            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
        peft_config.inference_mode = inference_mode

        # Save the base model as well
        self.base_model.save_pretrained(f'{save_directory}/base_model', safe_serialization)