"""端到端循环的长度上限与停止 token 边界测试。"""

import torch

from dflash_specblock.engine import DFlashSpecBlockEngine
from dflash_specblock.tree import DraftTree
from dflash_specblock.verification import GreedyPath, VerificationResult


class _Cache:
    def get_seq_length(self) -> int:
        return 1


class _Adapter:
    def propose_first(self, **_: object) -> object:
        return object()

    def propose_continuation(self, *_: object) -> object:
        raise AssertionError("空测试树不应扩展后续 block")


class _TreeBuilder:
    block_size = 4
    tree_budget = 16

    def build(self, *args: object, **kwargs: object) -> DraftTree:
        return DraftTree()


class _Verifier:
    def __init__(self, path: GreedyPath) -> None:
        self.path = path

    def verify(self, **kwargs: object) -> VerificationResult:
        return VerificationResult(
            path=self.path,
            cache=kwargs["cache"],
            target_context=torch.zeros(1, 1, 4),
        )


class _Engine(DFlashSpecBlockEngine):
    def __init__(self, path: GreedyPath) -> None:
        super().__init__(
            target=torch.nn.Identity(),
            adapter=_Adapter(),
            tree_builder=_TreeBuilder(),
            verifier=_Verifier(path),
            device=torch.device("cpu"),
        )

    def _prefill(self, input_ids: torch.Tensor) -> tuple[int, object, torch.Tensor, float]:
        del input_ids
        return 1, _Cache(), torch.zeros(1, 1, 4), 0.0


def test_stop_token_is_committed_once_and_ends_generation() -> None:
    path = GreedyPath(node_indices=[0, 1], token_ids=[7, 9], bonus_token_id=10)
    result = _Engine(path).generate(
        torch.tensor([[3]]),
        max_new_tokens=8,
        stop_token_ids={9},
    )
    assert result.generated_ids.tolist() == [[1, 7, 9]]
    assert result.iterations[0].committed_tokens == 2


def test_stop_outside_remaining_length_does_not_trigger_false_stop() -> None:
    path = GreedyPath(node_indices=[0], token_ids=[7], bonus_token_id=9)
    result = _Engine(path).generate(
        torch.tensor([[3]]),
        max_new_tokens=2,
        stop_token_ids={9},
    )
    assert result.generated_ids.tolist() == [[1, 7]]
