"""实验配置。

配置对象只使用 Python 标准库，避免为了读取一个 JSON 文件引入额外运行时依赖。所有相对
路径都相对于配置文件所在工程目录解析，便于将整个目录复制到 NVIDIA GPU 服务器后运行。
"""

from __future__ import annotations

import json
import math
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
    # temperature>0 时不再使用 greedy/普通 DDTree 叶验证，而是从完整 DFlash
    # slot 分布采样 gbv_paths 条路径，合并成树后执行分布保持的 GBV。
    temperature: float = 1.0
    gbv_paths: int = 3
    # 草稿树拓扑：``specblock`` 为 rank-guided 主链 + 兄弟分支（可跨块）；
    # ``ddtree`` 为 DDTree 官方的全局 best-first 堆式分配（单块、无需 rank head）。
    tree_mode: str = "specblock"
    # 仅 ddtree 模式生效的可选改动：预留 greedy 链，默认关闭以严格复现官方。
    ddtree_reserve_greedy_chain: bool = False
    # latency-aware DDTree 在一次最大树枚举中选择的嵌套预算。它以当前 proposal 的
    # prefix probability mass 预测接受长度，并在线学习当前 GPU 的 verify 延迟。
    ddtree_budget_candidates: tuple[int, ...] = (30, 45, 60, 80, 100, 128)
    ddtree_initial_budget: int = 60
    ddtree_warmup_rounds_per_budget: int = 1
    ddtree_policy_ewma_alpha: float = 0.2
    ddtree_exploration_interval: int = 64
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
        if "ddtree_budget_candidates" in raw:
            raw["ddtree_budget_candidates"] = tuple(
                int(x) for x in raw["ddtree_budget_candidates"]
            )
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
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature 必须是有限的非负数")
        if self.gbv_paths < 1:
            raise ValueError("gbv_paths 必须为正整数")
        if self.tree_mode not in {"specblock", "ddtree", "ddtree_adaptive"}:
            raise ValueError("tree_mode 仅支持 specblock、ddtree 或 ddtree_adaptive")
        if not isinstance(self.ddtree_reserve_greedy_chain, bool):
            raise ValueError("ddtree_reserve_greedy_chain 必须是布尔值")
        if self.temperature == 0 and self.tree_mode in {"ddtree", "ddtree_adaptive"}:
            # DDTree 的全部候选都来自同一次 block forward，跨块 continuation 无从定义。
            if self.max_blocks != 1:
                raise ValueError("tree_mode=ddtree 是单块方法，max_blocks 必须为 1")
            # DDTree 用 log-prob 做全局预算分配，不读取 rank head 的输出。
            if self.rank_mode != "heuristic":
                raise ValueError(
                    "tree_mode=ddtree 不使用 rank head，rank_mode 必须为 heuristic"
                )
            if self.rank_checkpoint:
                raise ValueError("tree_mode=ddtree 不加载 rank checkpoint，请设为 null")
        elif self.temperature == 0 and self.ddtree_reserve_greedy_chain:
            raise ValueError("ddtree_reserve_greedy_chain 仅在 tree_mode=ddtree 下有意义")
        if self.temperature == 0 and self.tree_mode == "ddtree_adaptive":
            candidates = tuple(int(value) for value in self.ddtree_budget_candidates)
            if not candidates or tuple(sorted(set(candidates))) != candidates:
                raise ValueError("ddtree_budget_candidates 必须是严格递增且非空的整数序列")
            if candidates[0] < self.block_size or candidates[-1] > self.tree_budget:
                raise ValueError(
                    "ddtree_budget_candidates 必须位于 [block_size, tree_budget]"
                )
            if self.ddtree_initial_budget not in candidates:
                raise ValueError("ddtree_initial_budget 必须属于候选预算")
            if self.ddtree_warmup_rounds_per_budget < 1:
                raise ValueError("ddtree_warmup_rounds_per_budget 必须至少为 1")
            if not 0.0 < self.ddtree_policy_ewma_alpha <= 1.0:
                raise ValueError("ddtree_policy_ewma_alpha 必须位于 (0, 1]")
            if self.ddtree_exploration_interval < 0:
                raise ValueError("ddtree_exploration_interval 不能为负数")
            if self.ddtree_reserve_greedy_chain:
                raise ValueError("ddtree_adaptive 不与 reserve_greedy_chain 叠加")
            if self.use_cuda_graphs:
                raise ValueError(
                    "ddtree_adaptive 需要可变 verify shape，不能使用固定最大形状 CUDA Graph"
                )
        if self.rank_mode not in {"heuristic", "learned"}:
            raise ValueError("rank_mode 仅支持 heuristic 或 learned")
        if self.temperature == 0 and self.rank_mode == "learned" and not self.rank_checkpoint:
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
