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


def load_models(config: ExperimentConfig, device: torch.device) -> ModelBundle:
    """显式 `.to(npu)`，不使用其他后端的自动设备分片路径。"""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_name(config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.target_path,
        revision=config.target_revision,
        trust_remote_code=True,
    )

    # eager attention 能可靠接收验证器生成的 4D ancestor-only additive mask。
    target = AutoModelForCausalLM.from_pretrained(
        config.target_path,
        revision=config.target_revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    draft = AutoModel.from_pretrained(
        config.draft_path,
        revision=config.draft_revision,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()

    target.to(device)
    draft.to(device)
    _validate_model_pair(target, draft)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    for parameter in draft.parameters():
        parameter.requires_grad_(False)
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
