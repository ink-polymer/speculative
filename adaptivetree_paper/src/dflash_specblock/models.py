"""Qwen3 目标模型与 DFlash 官方 checkpoint 的加载入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .device import dtype_from_name


@dataclass(slots=True)
class ModelBundle:
    target: torch.nn.Module
    draft: torch.nn.Module
    tokenizer: object


@dataclass(slots=True)
class TargetModelBundle:
    target: torch.nn.Module
    tokenizer: object


def _validate_model_pair(target: torch.nn.Module, draft: torch.nn.Module) -> None:
    """在进入解码前验证 DFlash checkpoint 与 Qwen target 的结构契约。"""
    required_draft = ("target_layer_ids", "fc", "hidden_norm", "layers", "rotary_emb")
    missing = [name for name in required_draft if not hasattr(draft, name)]
    if missing:
        raise TypeError(f"加载到的 draft 不是兼容的 DFlash 模型，缺少: {missing}")

    target_hidden = int(target.config.hidden_size)
    draft_hidden = int(draft.config.hidden_size)
    if target_hidden != draft_hidden:
        raise ValueError(f"target hidden={target_hidden} 与 DFlash hidden={draft_hidden} 不一致")

    target_layers = int(target.config.num_hidden_layers)
    declared_target_layers = int(getattr(draft.config, "num_target_layers", -1))
    if declared_target_layers != target_layers:
        raise ValueError(
            "DFlash 声明的 target 层数与实际模型不一致: "
            f"draft={declared_target_layers}, target={target_layers}"
        )
    invalid_layers = [
        int(index)
        for index in draft.target_layer_ids
        if not 0 <= int(index) < target_layers
    ]
    if invalid_layers:
        raise ValueError(f"DFlash target_layer_ids 越界: {invalid_layers}")

    lm_head = target.get_output_embeddings()
    target_vocab = int(target.config.vocab_size)
    draft_vocab = int(draft.config.vocab_size)
    if target_vocab != draft_vocab:
        raise ValueError(f"target vocab={target_vocab} 与 DFlash vocab={draft_vocab} 不一致")
    if (
        lm_head is None
        or int(lm_head.weight.shape[0]) != target_vocab
        or int(lm_head.weight.shape[1]) != draft_hidden
    ):
        raise ValueError("target LM head 与 DFlash hidden 维度不兼容")

    expected_fc_input = len(draft.target_layer_ids) * target_hidden
    if int(draft.fc.in_features) != expected_fc_input or int(draft.fc.out_features) != draft_hidden:
        raise ValueError("DFlash target feature projection 的输入/输出维度不符合官方结构")
    if len(draft.layers) != int(draft.config.num_hidden_layers):
        raise ValueError("DFlash decoder layer 数与 config.num_hidden_layers 不一致")


def _pretrained_kwargs(
    config: ExperimentConfig,
    revision: str,
    device: torch.device,
    attn_implementation: str,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "revision": revision,
        "trust_remote_code": True,
        "dtype": dtype_from_name(config.dtype),
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if device.type == "cuda":
        # Load shards directly on the selected GPU instead of materializing a second
        # full copy on CPU and moving it afterwards.
        kwargs["device_map"] = {"": 0 if device.index is None else device.index}
    return kwargs


def _validate_cuda_precision(config: ExperimentConfig, device: torch.device) -> None:
    if device.type != "cuda" or config.dtype != "bfloat16":
        return
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    supported = False
    if callable(checker):
        with torch.cuda.device(device):
            supported = bool(checker())
    if not supported:
        raise RuntimeError(
            "bfloat16 was requested, but the selected NVIDIA GPU does not support CUDA BF16; "
            "use float16 (with AMP scaling for training) or a BF16-capable GPU"
        )


def _freeze_and_place(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.eval()
    if device.type != "cuda":
        model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _maybe_compile(
    model: torch.nn.Module,
    mode: str | None,
) -> torch.nn.Module:
    if mode is None:
        return model
    module_compile = getattr(model, "compile", None)
    if callable(module_compile):
        module_compile(mode=mode)
        return model
    return torch.compile(model, mode=mode)


def _load_target_and_tokenizer(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_cuda_precision(config, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.target_path,
        revision=config.target_revision,
        trust_remote_code=True,
    )
    target = AutoModelForCausalLM.from_pretrained(
        config.target_path,
        **_pretrained_kwargs(
            config,
            config.target_revision,
            device,
            config.attn_implementation,
        ),
    )
    return _freeze_and_place(target, device), tokenizer


def load_target_model(config: ExperimentConfig, device: torch.device) -> TargetModelBundle:
    """Load only the target and tokenizer for target-only generation workloads."""
    target, tokenizer = _load_target_and_tokenizer(config, device)
    target = _maybe_compile(target, config.torch_compile_mode)
    return TargetModelBundle(target=target, tokenizer=tokenizer)


def load_models(config: ExperimentConfig, device: torch.device) -> ModelBundle:
    """Load the target and DFlash draft directly onto an NVIDIA GPU when selected."""
    from transformers import AutoModel

    target, tokenizer = _load_target_and_tokenizer(config, device)

    draft = AutoModel.from_pretrained(
        config.draft_path,
        **_pretrained_kwargs(
            config,
            config.draft_revision,
            device,
            config.draft_attn_implementation or config.attn_implementation,
        ),
    )
    draft = _freeze_and_place(draft, device)

    _validate_model_pair(target, draft)
    target = _maybe_compile(target, config.torch_compile_mode)
    draft = _maybe_compile(draft, config.torch_compile_mode)
    return ModelBundle(target=target, draft=draft, tokenizer=tokenizer)


def render_prompt(tokenizer: object, prompt: str, enable_thinking: bool) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(return_tensors="pt", add_generation_prompt=True, tokenize=True)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
