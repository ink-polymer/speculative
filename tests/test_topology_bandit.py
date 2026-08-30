from __future__ import annotations

from pathlib import Path

from dflash_specblock.topology_bandit import (
    TopologyRatioBandit,
    prompt_context_features,
)


def _bandit(*, learning: bool) -> TopologyRatioBandit:
    return TopologyRatioBandit(
        actions=("ddtree:60", "dpv:2", "dpv:4"),
        initial_action="ddtree:60",
        ridge=1.0,
        exploration_scale=0.0,
        warmup_episodes_per_action=2,
        learning_enabled=learning,
        random_seed=7,
        policy_metadata={"temperature": "0.0"},
    )


def test_balanced_logging_and_frozen_roundtrip(tmp_path: Path) -> None:
    policy = _bandit(learning=True)
    context = prompt_context_features("Find integer 12.\ndef solve():", 12)
    for _ in range(24):
        action = policy.select(context)
        # Make dpv:2 the unambiguous throughput optimum.
        decode_ms = {"ddtree:60": 12.0, "dpv:2": 4.0, "dpv:4": 8.0}[action]
        policy.observe(committed_tokens=16, decode_ms=decode_ms)
    diagnostics = policy.diagnostics()
    assert min(diagnostics["observations"].values()) >= 2

    checkpoint = tmp_path / "policy.json"
    policy.save_policy(checkpoint)
    frozen = _bandit(learning=False)
    # Checkpoint configuration intentionally records the training mode only as
    # runtime state, so the same artifact can be loaded for frozen inference.
    frozen.load_policy(checkpoint)
    assert frozen.select(context) == "dpv:2"
    frozen.observe(committed_tokens=16, decode_ms=4.0)


def test_prompt_features_do_not_require_dataset_label() -> None:
    features = prompt_context_features("证明 2 + 2 = 4", 9)
    assert features.shape == (9,)
    assert features[0] == 1.0
