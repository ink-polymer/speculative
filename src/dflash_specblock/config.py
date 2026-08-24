"""实验配置。

配置对象只使用 Python 标准库，避免为了读取一个 JSON 文件引入额外运行时依赖。所有相对
路径都相对于配置文件所在工程目录解析，便于将整个目录复制到 NVIDIA GPU 服务器后运行。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    target_model_id: str = "Qwen/Qwen3-4B"
    draft_model_id: str = "z-lab/Qwen3-4B-DFlash-b16"
    target_revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    draft_revision: str = "b74e3a329c4d963783143b1e970d95b002be72bd"
    target_local_dir: str = "models/Qwen3-4B"
    draft_local_dir: str = "models/Qwen3-4B-DFlash-b16"
    # 正式配置 fail-closed：显式请求 CUDA 时不可用会直接报错，不静默切到 CPU。
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    draft_attn_implementation: str | None = None
    allow_tf32: bool = True
    torch_compile_mode: str | None = None
    use_cuda_graphs: bool = False
    cuda_graph_max_cache_len: int = 4096
    block_size: int = 4
    max_blocks: int = 2
    tree_budget: int = 60
    beam_width: int = 10
    branch_factors: tuple[int, int, int, int] = (2, 4, 10, 0)
    rank_mode: str = "learned"
    rank_checkpoint: str | None = "checkpoints/rank_head.pt"
    max_new_tokens: int = 128
    enable_thinking: bool = False
    seed: int = 42
    project_root: Path = Path(".")

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            raw: dict[str, Any] = json.load(stream)

        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"配置包含未知字段: {unknown}")
        if "branch_factors" in raw:
            raw["branch_factors"] = tuple(int(x) for x in raw["branch_factors"])
        raw["project_root"] = config_path.parent.parent
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 1 <= self.block_size <= 15:
            raise ValueError("block_size 必须在 [1, 15]；DFlash-b16 除锚点外最多预测 15 个位置")
        if self.max_blocks < 1:
            raise ValueError("max_blocks 必须至少为 1")
        if self.tree_budget < self.block_size:
            raise ValueError("tree_budget 不能小于 block_size")
        if self.beam_width < 1:
            raise ValueError("beam_width 必须为正整数")
        if len(self.branch_factors) != 4 or any(x < 0 for x in self.branch_factors):
            raise ValueError("branch_factors 必须是四个非负整数")
        if self.rank_mode not in {"heuristic", "learned"}:
            raise ValueError("rank_mode 仅支持 heuristic 或 learned")
        if self.rank_mode == "learned" and not self.rank_checkpoint:
            raise ValueError("learned 模式必须配置 rank_checkpoint")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype 仅支持 bfloat16、float16 或 float32")
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("target attn_implementation 仅支持 eager 或 sdpa，以兼容 4D tree mask")
        supported_draft_attention = {
            None,
            "eager",
            "sdpa",
            "flash_attention_2",
            "flash_attention_3",
        }
        if self.draft_attn_implementation not in supported_draft_attention:
            raise ValueError(
                "draft_attn_implementation 仅支持 null、eager、sdpa、"
                "flash_attention_2 或 flash_attention_3"
            )
        if (
            self.draft_attn_implementation is not None
            and self.draft_attn_implementation.startswith("flash_attention_")
            and self.dtype == "float32"
        ):
            raise ValueError("draft FlashAttention 需要 float16 或 bfloat16 dtype")
        if not isinstance(self.allow_tf32, bool):
            raise ValueError("allow_tf32 必须是布尔值")
        supported_compile_modes = {
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        if self.torch_compile_mode not in supported_compile_modes:
            raise ValueError(
                "torch_compile_mode 必须为 null、default、reduce-overhead、"
                "max-autotune 或 max-autotune-no-cudagraphs"
            )
        if not isinstance(self.use_cuda_graphs, bool):
            raise ValueError("use_cuda_graphs 必须是布尔值")
        if self.cuda_graph_max_cache_len < 1:
            raise ValueError("cuda_graph_max_cache_len 必须为正整数")
        compile_uses_cuda_graphs = self.torch_compile_mode in {
            "reduce-overhead",
            "max-autotune",
        }
        if self.use_cuda_graphs and compile_uses_cuda_graphs:
            raise ValueError(
                "use_cuda_graphs 不能与会启用 CUDA Graph 的 torch_compile_mode 嵌套"
            )
        normalized_device = self.device.lower()
        supported_device = normalized_device in {"auto", "cpu"} or bool(
            re.fullmatch(r"cuda(?::\d+)?", normalized_device)
        )
        if not supported_device:
            raise ValueError("device 仅支持 auto、cpu、cuda 或 cuda:<id>")
        if normalized_device == "cpu" and self.use_cuda_graphs:
            raise ValueError("CPU 配置不能启用 use_cuda_graphs")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须为正整数")

    def resolve_model_path(self, local_dir: str, remote_id: str) -> str:
        local = (self.project_root / local_dir).resolve()
        return str(local) if local.exists() else remote_id

    @property
    def target_path(self) -> str:
        return self.resolve_model_path(self.target_local_dir, self.target_model_id)

    @property
    def draft_path(self) -> str:
        return self.resolve_model_path(self.draft_local_dir, self.draft_model_id)

    @property
    def rank_checkpoint_path(self) -> Path | None:
        if not self.rank_checkpoint:
            return None
        value = Path(self.rank_checkpoint).expanduser()
        return value.resolve() if value.is_absolute() else (self.project_root / value).resolve()
