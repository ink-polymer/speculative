"""DFlash 扩散草稿与树验证的 NVIDIA CUDA 实现。

提供两种草稿树拓扑，共用同一套 DFlash drafter、ancestor-only 验证与 KV 压缩：

- ``SpecBlockTreeBuilder``：rank-guided 主链 + 兄弟分支，可跨块 continuation；
- ``DDTreeBuilder``：DDTree 官方的全局 best-first 预算分配，单块、无需 rank head；
- ``LatencyAwareDDTreeBuilder``：按当前 proposal 与实测 GPU 延迟自适应选择节点预算。
- ``ContextualBanditDDTreeBuilder``：用在线 LinUCB 直接学习吞吐最优的节点预算。
- ``PPODDTreeBuilder``：用离散 PPO 和端到端时间奖励学习节点预算，是主 RL 策略。
"""

from .bandit_builder import ContextualBanditDDTreeBuilder
from .config import ExperimentConfig
from .ddtree_builder import DDTreeBuilder, LatencyAwareDDTreeBuilder
from .engine import DFlashSpecBlockEngine, GenerationResult
from .ppo_builder import PPODDTreeBuilder
from .tree import BlockProposal, DraftTree, SpecBlockTreeBuilder

__all__ = [
    "BlockProposal",
    "ContextualBanditDDTreeBuilder",
    "DDTreeBuilder",
    "DFlashSpecBlockEngine",
    "DraftTree",
    "ExperimentConfig",
    "GenerationResult",
    "LatencyAwareDDTreeBuilder",
    "PPODDTreeBuilder",
    "SpecBlockTreeBuilder",
]
