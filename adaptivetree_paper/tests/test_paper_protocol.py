from __future__ import annotations

import itertools
import math
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dflash_specblock.paper.common import BASELINES, ROOT, atomic_json, contract, digest, file_hash, load_config, load_json, run_lock, verify_contract
from dflash_specblock.paper.data import SOURCES, make_row, normalize, partition_train, validate_rows, user_turns
from dflash_specblock.paper.evaluation import evaluate, measure_case, paired_bootstrap, same_dialogue, summarize
from dflash_specblock.paper.runtime import PaperRuntime, commit
from dflash_specblock.paper.structure import Action, FEATURE_DIM, K, StructurePolicy, build_layered, features, static_action
from dflash_specblock.paper.training import cf_variant, collect, episode_returns, policy_update, restore, train

torch.set_num_threads(1)


def paths(tree):
    result = []
    for node in tree.nodes:
        result.append((() if node.parent < 0 else result[node.parent]) + (node.token_id,))
    return result


def reference_tree(logits, action):
    # At a fixed depth, all paths have the same sum of log normalizers.
    # Raw integer logit sums provide an exact, independent reference for ties.
    logp = logits.double().tolist()
    kept, previous = [], [()]
    for d in range(action.depth):
        remaining = action.budget - len(kept)
        if not remaining:
            break
        top = sorted(range(len(logp[d])), key=lambda t: (-logp[d][t], t))[:action.widths[d]]
        candidates = [parent + (token,) for parent in previous for token in top]
        candidates.sort(key=lambda p: (-sum(logp[i][t] for i, t in enumerate(p)), p))
        previous = candidates[:min(action.quotas[d], remaining)]
        kept += previous
    return kept


@pytest.mark.parametrize("seed", range(5))
def test_layered_matches_independent_recurrence(seed):
    rng = random.Random(seed)
    for _ in range(150):
        depth = rng.randrange(5)
        logits = torch.tensor([[rng.randrange(-3, 4) for _ in range(6)] for _ in range(4)], dtype=torch.float32)
        action = Action(rng.randrange(20), depth, tuple(rng.randrange(8) for _ in range(depth)),
                        tuple(rng.randrange(1, 7) for _ in range(depth)))
        tree = build_layered(logits, action)
        assert paths(tree) == reference_tree(logits, action)
        assert len(tree) <= action.budget
        assert all(not p[:-1] or p[:-1] in paths(tree) for p in paths(tree))


def test_cutoff_ties_choose_lowest_token_ids():
    tree = build_layered(torch.zeros(3, 100), Action(20, 3, (4, 8, 8), (4, 4, 4)))
    assert paths(tree) == reference_tree(torch.zeros(3, 100), Action(20, 3, (4, 8, 8), (4, 4, 4)))


@pytest.mark.parametrize("action", [Action(-1, 0, (), ()), Action(3, 1, (), (2,)),
                                    Action(3, 1, (2,), (0,)), Action(1.5, 0, (), ())])
def test_illegal_actions_rejected(action):
    with pytest.raises(ValueError):
        build_layered(torch.ones(2, 3), action)


def test_all_ablations_and_inactive_head_probabilities():
    from dflash_specblock.paper.common import VARIANTS
    x = torch.randn(FEATURE_DIM)
    rng = random.Random(2)
    for variant in VARIANTS:
        policy = StructurePolicy(variant)
        action, indices = policy.random_action(rng)
        assert len(build_layered(torch.randn(K, 32), action)) <= action.budget
        if variant == "fixed_budget": assert action.budget == 60
        if variant == "fixed_depth": assert action.depth == 15
        if variant == "fixed_quotas": assert set(action.quotas) == {4}
        if variant == "fixed_width": assert set(action.widths) == {4}
        logp, value = policy.log_prob_value(x, torch.tensor(indices))
        assert torch.isfinite(logp).all() and torch.isfinite(value).all()
        indices[1] = 0  # shortest available depth
        action = policy.decode(indices)
        before, _ = policy.log_prob_value(x, torch.tensor(indices))
        for d in range(action.depth, K):
            indices[2 + d] = (indices[2 + d] + 1) % len(policy.choices[2 + d])
            indices[2 + K + d] = (indices[2 + K + d] + 1) % len(policy.choices[2 + K + d])
        after, _ = policy.log_prob_value(x, torch.tensor(indices))
        assert torch.equal(before, after)


@pytest.mark.parametrize("variant,slice_", [("no_target", slice(60, 76)), ("no_history", slice(76, 82)), ("draft_only", slice(60, 82))])
def test_feature_ablations_really_remove_information(variant, slice_):
    policy = StructurePolicy(variant)
    x = torch.randn(FEATURE_DIM)
    y = x.clone()
    y[slice_] += 1000
    assert all(torch.equal(a, b) for a, b in zip(policy(x)[0], policy(y)[0]))


def test_rewards_equal_full_latency_and_update_policy():
    policy = StructurePolicy()
    x = torch.randn(FEATURE_DIM)
    action, indices = policy.choose(x, sample=True)
    result = {"wall_ms": 50., "rounds": [
        {"latency_ms": 10., "accepted": 2, "committed": 3, "features": x.tolist(), "indices": indices},
        {"latency_ms": 20., "accepted": 0, "committed": 1, "features": x.tolist(), "indices": indices},
    ]}
    returns = episode_returns(result, "full")
    assert float(returns[0]) == pytest.approx(-0.05)
    assert returns.tolist() == pytest.approx([-0.05, -0.04])
    previous = policy.actor.weight.detach().clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    policy_update(policy, optimizer, result, 0.5)
    assert not torch.equal(previous, policy.actor.weight)


def test_full_counts_and_no_test_leakage():
    assert {k: v[3] for k, v in SOURCES.items()} == {
        "gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 500, "mbpp_sanitized": 257,
        "aime25": 30, "livecodebench": 1055, "mt-bench": 80}
    training = [make_row("gsm8k", {"question": f"question {i}"}, i, "train", "source", "main") for i in range(100)]
    evaluation = [make_row("gsm8k", {"question": "question 2"}, 2, "test", "source", "main")]
    train_rows, dev, removed = partition_train(training, evaluation, 0.2)
    assert len(train_rows) + len(dev) + len(removed) == 100
    assert len(removed) == 1
    assert not ({r["prompt_hash"] for r in train_rows} & {r["prompt_hash"] for r in dev + evaluation})
    assert normalize("Ａ B\n C") == normalize("a b c")


def test_resume_fails_on_changed_contract(tmp_path):
    contract(tmp_path, {"data": "old", "seed": 1})
    with pytest.raises(ValueError):
        contract(tmp_path, {"data": "new", "seed": 1})


def test_readonly_contract_and_concurrent_run_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_contract(tmp_path, {"fixture": True})
    assert not (tmp_path / "contract.json").exists()
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass


@pytest.mark.parametrize("changes", [{"variants": ["full", "full"]}, {"cf_repeats": 0},
    {"eval_repeats": 0}, {"oracle_budgets": [200, 400]}, {"gamma": .99},
    {"temperature": 1}, {"warmup_runs": 0}, {"seeds": [True]}, {"validation_fraction": 1.}])
def test_invalid_protocol_rejected(tmp_path, changes):
    cfg = load_json(ROOT / "configs/paper_t0_full.json")
    atomic_json(tmp_path / "config.json", {**cfg, **changes})
    with pytest.raises(ValueError):
        load_config(tmp_path / "config.json")


def test_recompute_data_identity_and_official_split_roles():
    training = [make_row("gsm8k", {"question": f"train {i}"}, i, "train", "openai/gsm8k", "main") for i in range(80)]
    evaluation = [make_row("gsm8k", {"question": "holdout"}, 0, "test", "openai/gsm8k", "main")]
    train_rows, dev, _ = partition_train(training, evaluation, .2)
    validate_rows(train_rows, dev, evaluation, .2)
    tampered = [{**train_rows[0], "prompt": "different text"}, *train_rows[1:]]
    with pytest.raises(ValueError, match="identity or hash"):
        validate_rows(tampered, dev, evaluation, .2)
    bad = make_row("gsm8k", train_rows[0]["reference"], train_rows[0]["source_index"],
                   "test", "openai/gsm8k", "main")
    with pytest.raises(ValueError, match="official training"):
        validate_rows([bad, *train_rows[1:]], dev, evaluation, .2)
    with pytest.raises(ValueError, match="partition changed"):
        validate_rows(dev, train_rows, evaluation, .2)


def test_publication_rejects_partial_data_even_if_not_marked_smoke(tmp_path):
    with pytest.raises(ValueError, match="every full official"):
        summarize(tmp_path, [], load_config(ROOT / "configs/paper_t0_full.json"))


@pytest.mark.parametrize("smoke_count", [0, 2])
def test_loader_refuses_partial_source_even_in_smoke(tmp_path, monkeypatch, smoke_count):
    from dflash_specblock.paper import __main__ as cli
    row = make_row("gsm8k", {"question": "one row is not full"}, 0,
                   "test", "openai/gsm8k", "main")
    monkeypatch.setattr(cli, "check_manifest", lambda _: {})
    monkeypatch.setattr(cli, "read_rows", lambda _: [row])
    with pytest.raises(ValueError, match="Not a full official test split"):
        cli.load_data(tmp_path, smoke_count)


def test_stopping_and_paired_aggregation():
    for cap, eos in itertools.product(range(5), ({2}, {5}, set())):
        out = commit([1, 2, 3, 4], cap, eos)
        assert len(out) <= cap
        assert not any(x in eos for x in out[:-1])
    assert paired_bootstrap([2., 20.], [1., 10.], 100) == [2., 2.]


class TinyTarget(nn.Module):
    """A real two-layer causal attention network and real Transformers DynamicCache."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(618)
        self.embedding = nn.Embedding(32, 8).double()
        self.head = nn.Linear(8, 32).double()
        self.qkv = nn.ModuleList([nn.Linear(8, 24).double() for _ in range(2)])
        self.config = SimpleNamespace(max_position_embeddings=10000)
        self.generation_config = SimpleNamespace(eos_token_id=None)
    def get_input_embeddings(self): return self.embedding
    def get_output_embeddings(self): return self.head
    def forward(self, input_ids, past_key_values, attention_mask=None, position_ids=None, logits_to_keep=None, **kwargs):
        past = int(past_key_values.get_seq_length())
        n = input_ids.shape[1]
        positions = torch.arange(past, past + n)[None] if position_ids is None else position_ids
        h = self.embedding(input_ids) + torch.sin(positions[..., None].double() / 17) / 10
        states = [h]
        if attention_mask is None:
            allowed = torch.arange(past + n)[None, :] <= torch.arange(past, past + n)[:, None]
            attention_mask = torch.where(allowed, 0., -torch.inf)[None, None]
        for layer, projection in enumerate(self.qkv):
            q, k, v = projection(h).chunk(3, -1)
            k, v = past_key_values.update(k[:, None], v[:, None], layer)
            weights = (q[:, None] @ k.transpose(-2, -1) / math.sqrt(8) + attention_mask).softmax(-1)
            h = torch.tanh(h + (weights @ v).squeeze(1))
            states.append(h)
        logits = self.head(h)
        if logits_to_keep: logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, hidden_states=tuple(states), past_key_values=past_key_values)


class TinyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, block_size=16, dflash_config={"mask_token_id": 31})
        self.block_size, self.target_layer_ids = 16, [0]
        self.fc = self.hidden_norm = self.norm = nn.Identity()
        self.layers = nn.ModuleList([nn.Identity()])
    def rotary_emb(self, *args): return None
    def forward(self, target_hidden, noise_embedding, past_key_values, **kwargs):
        combined = torch.cat((target_hidden, noise_embedding), 1)
        past_key_values.update(combined[:, None], combined[:, None], 0)
        return noise_embedding + target_hidden.mean(1, keepdim=True) * .1


def tiny_runtime():
    pytest.importorskip("transformers")
    from dflash_specblock.ddtree_builder import DDTreeBuilder
    from dflash_specblock.dflash_adapter import DFlashBlockAdapter
    from dflash_specblock.rank_head import HeuristicRanker
    from dflash_specblock.vanilla_engine import VanillaDFlashEngine
    from dflash_specblock.verification import TargetTreeVerifier
    runtime = object.__new__(PaperRuntime)
    runtime.cfg = {**load_config(ROOT / "configs/paper_t0_full.json"), "max_new_tokens": 6,
        "train_epochs": 1, "pretrain_epochs": 1, "cf_actions": 2, "cf_repeats": 1,
        "warmup_runs": 0, "eval_repeats": 1, "bootstrap_samples": 20,
        "seeds": [17], "variants": ["full", "no_online_rl", "no_pretrain"]}
    runtime.device = torch.device("cpu")
    runtime.target = TinyTarget().eval()
    runtime.adapter = DFlashBlockAdapter(runtime.target, TinyDraft(), HeuristicRanker(), K)
    runtime.verifier = TargetTreeVerifier(runtime.target, [0], runtime.device, torch.float64)
    runtime.ddtree = DDTreeBuilder(K, 60)
    runtime.vanilla = VanillaDFlashEngine(runtime.target, runtime.adapter, runtime.device, torch.float64)
    runtime.stops = set()
    runtime.model_cfg = SimpleNamespace(enable_thinking=False)
    runtime.tokenizer = SimpleNamespace(decode=lambda tokens, **_: " ".join(map(str, tokens)),
        apply_chat_template=lambda messages, **_: torch.tensor([[1] + [sum(map(ord, m["content"])) % 32 for m in messages]]))
    runtime.encode = lambda _: torch.tensor([[1, 3]])
    return runtime


def test_actual_attention_runtime_greedy_and_counterfactuals():
    runtime = tiny_runtime()
    ids = runtime.encode("test")
    reference = runtime.generate(ids, "ar")["tokens"]
    for method in BASELINES:
        assert runtime.generate(ids, method)["tokens"] == reference
    for variant in ("full", "fixed_budget", "no_target"):
        policy = StructurePolicy(variant)
        observed = []
        def observe(state):
            original = int(state.cache.get_seq_length())
            observed.append(runtime.counterfactuals(state, policy, random.Random(1)))
            assert state.cache.get_seq_length() == original
        result = runtime.generate(ids, "layered", policy, observer=observe)
        assert result["tokens"] == reference
        assert observed


@pytest.mark.parametrize("cap", [0, 1, 2, 7, 18])
def test_real_cache_stopping_and_stochastic_structure(cap):
    runtime = tiny_runtime()
    runtime.cfg["max_new_tokens"] = cap
    ids = runtime.encode("test")
    reference = runtime.generate(ids, "ar")["tokens"]
    for stops in (set(), set(reference[:1]), set(reference[2:3])):
        runtime.stops = stops
        expected = runtime.generate(ids, "ar")["tokens"]
        for method in BASELINES:
            assert runtime.generate(ids, method)["tokens"] == expected
        assert runtime.generate(ids, "layered", StructurePolicy(), sample=True,
                                generator=torch.Generator().manual_seed(cap))["tokens"] == expected


def test_transformers_qwen3_attention_mask_and_kv_integration():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from dflash_specblock.dflash_adapter import DFlashBlockAdapter
    from dflash_specblock.rank_head import HeuristicRanker
    from dflash_specblock.vanilla_engine import VanillaDFlashEngine
    from dflash_specblock.verification import TargetTreeVerifier
    runtime = tiny_runtime()
    runtime.cfg["max_new_tokens"] = 18
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=256)
    config._attn_implementation = "eager"
    runtime.target = Qwen3ForCausalLM(config).double().eval()
    draft = TinyDraft()
    draft.config.hidden_size = 16
    runtime.adapter = DFlashBlockAdapter(runtime.target, draft, HeuristicRanker(), K)
    runtime.verifier = TargetTreeVerifier(runtime.target, [0], runtime.device, torch.float64)
    runtime.vanilla = VanillaDFlashEngine(runtime.target, runtime.adapter, runtime.device, torch.float64)
    ids = runtime.encode("test")
    reference = runtime.generate(ids, "ar")["tokens"]
    for method in BASELINES:
        assert runtime.generate(ids, method)["tokens"] == reference
    policy = StructurePolicy()
    assert runtime.generate(ids, "layered", policy, sample=True,
                            generator=torch.Generator().manual_seed(5))["tokens"] == reference


def test_full_local_pipeline_and_resume(tmp_path):
    runtime = tiny_runtime()
    rows = [make_row("gsm8k", {"question": f"question {i}"}, i, "train", "source", "main") for i in range(2)]
    dev = [make_row("gsm8k", {"question": "dev"}, 3, "train", "source", "main")]
    cf = tmp_path / "counterfactual/full"
    collect(runtime, rows, cf, {"fixture": True}, "full", 17)
    checkpoints = {}
    for variant in runtime.cfg["variants"]:
        directory = tmp_path / "training/17" / variant
        checkpoints[variant] = train(runtime, rows, dev, cf, directory, {"fixture": True}, variant, 17)
        assert train(runtime, rows, dev, cf, directory, {"fixture": True}, variant, 17) == checkpoints[variant]
    evaluation_rows = dev + [make_row("mt-bench", {"question_id": 101, "category": "writing",
        "turns": ["Write a story.", "Explain your previous story."], "reference": ["unused", "unused"]},
        0, "test", "lm-sys/FastChat", None)]
    evaluate(runtime, evaluation_rows, checkpoints, tmp_path / "evaluation/17", {"fixture": True}, 17)
    evaluate(runtime, evaluation_rows, checkpoints, tmp_path / "evaluation/17", {"fixture": True}, 17)
    result = summarize(tmp_path, evaluation_rows, runtime.cfg, smoke=True)
    assert not result["publication_eligible"]
    assert len(result["rows"]) == 2 * (len(BASELINES) + len(runtime.cfg["variants"]))
    assert all(r["turns_per_prompt"] == 2 and r["measured_turns"] == 2
               for r in result["rows"] if r["dataset"] == "mt-bench")
    with pytest.raises(ValueError, match="Missing or unexpected"):
        evaluate(runtime, dev, {}, tmp_path / "bad_evaluation", {"fixture": True}, 17)
    with pytest.raises(ValueError, match="contract mismatch"):
        evaluate(runtime, dev, checkpoints, tmp_path / "bad_evaluation", {"fixture": False}, 17)
    path = tmp_path / "evaluation/17/prompts/000000.json"
    record = load_json(path)
    record["observations"]["ar"][0]["wall_ms"] = 1.
    atomic_json(path, record)
    with pytest.raises(ValueError, match="Changed or missing"):
        summarize(tmp_path, evaluation_rows, runtime.cfg, smoke=True)
    # Even a newly checksummed record cannot pass by lying about exact_match.
    record["observations"]["ar"][0]["wall_ms"] = sum(t["wall_ms"] for t in record["observations"]["ar"][0]["turns"])
    record["observations"]["full"][0]["tokens"][0] += 1
    atomic_json(path, record)
    completion_path = tmp_path / "evaluation/17/complete.json"
    completion = load_json(completion_path)
    completion["files"]["prompts/000000.json"] = file_hash(path)
    atomic_json(completion_path, completion)
    with pytest.raises(ValueError, match="differs from AR"):
        summarize(tmp_path, evaluation_rows, runtime.cfg, smoke=True)
