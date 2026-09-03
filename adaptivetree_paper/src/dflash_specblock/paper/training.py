"""Same-state counterfactual pretraining + episodic Monte Carlo actor-critic.

The target/draft are frozen. No PPO clipping, no discount, no test-set training.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import atomic_json, contract, file_hash, load_json, run_lock, verify_contract
from .structure import StructurePolicy


def cf_variant(variant):
    return variant if variant in {"fixed_budget", "fixed_depth", "fixed_quotas", "fixed_width"} else "full"


def training_metadata(metadata, variant, seed, cf_dir):
    return {**metadata, "stage": "train", "variant": variant, "seed": seed,
            "counterfactual_completion_sha256": None if variant == "no_pretrain"
            else file_hash(cf_dir / "complete.json")}


def checkpoint(path, policy, optimizer=None, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save({"policy": policy.state_dict(), "variant": policy.variant,
                "optimizer": optimizer.state_dict() if optimizer else None, **state}, temp)
    temp.replace(path)


def restore(path, policy, optimizer=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["variant"] != policy.variant:
        raise ValueError("Ablation checkpoint mismatch")
    policy.load_state_dict(data["policy"])
    if optimizer is not None and data.get("optimizer"):
        optimizer.load_state_dict(data["optimizer"])
    return data


def collect(runtime, rows, directory, metadata, variant, seed):
    policy = StructurePolicy(cf_variant(variant))
    with run_lock(directory):
        contract(directory, {**metadata, "stage": "counterfactual", "variant": policy.variant, "seed": seed})
        runtime.warmup([])
        for index, row in enumerate(rows):
            path = directory / "prompts" / f"{index:06d}.json"
            if path.exists():
                if load_json(path)["id"] != row["id"]:
                    raise ValueError("Counterfactual resume ID mismatch")
                continue
            rng = random.Random(f"{seed}:{row['id']}")
            states = []
            def observe(state):
                states.append(runtime.counterfactuals(state, policy, rng))
            result = runtime.generate(runtime.encode(row["prompt"]), "ddtree", observer=observe)
            atomic_json(path, {"id": row["id"], "dataset": row["dataset"],
                               "tokens": result["tokens"], "states": states})
            print(f"collect {policy.variant} {index + 1}/{len(rows)}", flush=True)
        files = [directory / "prompts" / f"{i:06d}.json" for i in range(len(rows))]
        c0 = l0 = best_c = best_nodes = 0.
        state_count = 0
        for path in files:
            for state in load_json(path)["states"]:
                candidates = state["candidates"]
                baseline = next(c for c in candidates if c["name"] == "ddtree_60")
                c0 += baseline["committed"]
                l0 += baseline["latency_ms"]
                best_c += max(c["committed"] for c in candidates)
                best_nodes += max(c["accepted"] for c in candidates)
                state_count += 1
        if not state_count or l0 <= 0:
            raise RuntimeError("No counterfactual states were collected")
        summary = {"prompts": len(rows), "states": state_count,
                   "rho_tokens_per_ms": c0 / l0,
                   "sampled_best_committed_ratio": best_c / c0,
                   "mean_sampled_best_accepted": best_nodes / state_count,
                   "not_a_true_global_oracle_or_speedup": True,
                   "files": {str(p.relative_to(directory)): file_hash(p) for p in files}}
        atomic_json(directory / "complete.json", summary)
        return summary


def checked_cf(directory, expected_ids):
    summary = load_json(directory / "complete.json")
    paths = [directory / "prompts" / f"{i:06d}.json" for i in range(len(expected_ids))]
    if len(summary["files"]) != len(paths):
        raise ValueError("Partial counterfactual dataset")
    for path, identity in zip(paths, expected_ids):
        if file_hash(path) != summary["files"].get(str(path.relative_to(directory))):
            raise ValueError(f"Counterfactual file changed: {path}")
        if load_json(path)["id"] != identity:
            raise ValueError("Counterfactual data is not this training split")
    return paths, summary


def paired_score(candidate, baseline, rho, variant):
    if variant == "acceptance_reward":
        return candidate["committed"] - baseline["committed"]
    if variant == "local_ratio_reward":
        return candidate["committed"] / candidate["latency_ms"]
    return ((candidate["committed"] - baseline["committed"])
            - rho * (candidate["latency_ms"] - baseline["latency_ms"]))


def pretrain(policy, paths, rho, cfg, seed):
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
    rng = random.Random(seed)
    for epoch in range(cfg["pretrain_epochs"]):
        order = list(paths)
        rng.shuffle(order)
        for path in order:
            states = load_json(path)["states"]
            rng.shuffle(states)
            for start in range(0, len(states), 64):
                chunk = states[start:start + 64]
                if not chunk:
                    continue
                xs, ys = [], []
                for state in chunk:
                    baseline = next(c for c in state["candidates"] if c["name"] == "ddtree_60")
                    valid = [c for c in state["candidates"] if c["indices"] is not None]
                    winner = max(valid, key=lambda c: paired_score(c, baseline, rho, policy.variant))
                    xs.append(state["features"])
                    ys.append(winner["indices"])
                logp, _ = policy.log_prob_value(torch.tensor(xs), torch.tensor(ys))
                loss = -logp.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step()
        print(f"pretrain {policy.variant} seed={seed} epoch={epoch + 1}", flush=True)


def episode_returns(result, variant):
    rounds = result["rounds"]
    if not rounds:
        return torch.empty(0)
    if variant == "acceptance_reward":
        rewards = [r["accepted"] for r in rounds]
    elif variant == "local_ratio_reward":
        rewards = [r["committed"] / max(r["latency_ms"] / 1000, 1e-9) for r in rounds]
    else:
        costs = [r["latency_ms"] / 1000 for r in rounds]
        # Include prefill, Python accounting, synchronization and the final tensor.
        # Sum of rewards is exactly negative measured end-to-end generation time.
        costs[-1] += result["wall_ms"] / 1000 - sum(costs)
        rewards = [-c for c in costs]
    return torch.tensor(list(reversed(list(_reverse_sums(rewards)))))


def _reverse_sums(values):
    total = 0.
    for value in reversed(values):
        total += value
        yield total


def policy_update(policy, optimizer, result, value_coef):
    if not result["rounds"]:
        return {"loss": 0., "return": 0., "rounds": 0}
    x = torch.tensor([r["features"] for r in result["rounds"]])
    indices = torch.tensor([r["indices"] for r in result["rounds"]])
    returns = episode_returns(result, policy.variant)
    logp, values = policy.log_prob_value(x, indices)
    advantage = returns - values.detach()
    # A sum over a complete episode, NOT mean-over-variable-number-of-rounds.
    actor_loss = -(logp * advantage).sum()
    value_loss = F.mse_loss(values, returns)
    loss = actor_loss + value_coef * value_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite policy loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    optimizer.step()
    return {"loss": float(loss.detach()), "return": float(returns[0]), "rounds": len(returns)}


def validate_policy(runtime, policy, rows):
    total = baseline_total = 0.
    for row in rows:
        ids = runtime.encode(row["prompt"])
        baseline = runtime.generate(ids, "ar")
        result = runtime.generate(ids, "layered", policy)
        if baseline["tokens"] != result["tokens"]:
            raise RuntimeError(f"Development greedy mismatch: {row['id']}")
        total += result["wall_ms"]
        baseline_total += baseline["wall_ms"]
    return {"wall_ms": total, "ar_wall_ms": baseline_total, "prompts": len(rows),
            "speedup": baseline_total / total}


def train(runtime, rows, dev_rows, cf_dir, directory, metadata, variant, seed):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    policy = StructurePolicy(variant)
    with run_lock(directory):
        identity = contract(directory, training_metadata(metadata, variant, seed, cf_dir))
        latest = directory / "latest.pt"
        optimizer = torch.optim.Adam(policy.parameters(), lr=runtime.cfg["learning_rate"])
        if (directory / "complete.json").exists():
            completion = load_json(directory / "complete.json")
            if completion["identity"] != identity or completion["best_sha256"] != file_hash(directory / "best.pt"):
                raise ValueError("Best checkpoint changed after training")
            return directory / "best.pt"
        if latest.exists():
            saved = restore(latest, policy, optimizer)
            if saved["identity"] != identity:
                raise ValueError("Checkpoint contract mismatch")
            generator.set_state(saved["sampling_rng"])
            torch.set_rng_state(saved["torch_rng"])
            epoch, offset, best = saved["epoch"], saved["offset"], saved["best"]
        else:
            if variant != "no_pretrain":
                verify_contract(cf_dir, {**metadata, "stage": "counterfactual", "variant": cf_variant(variant),
                                         "seed": runtime.cfg["seeds"][0]})
                paths, summary = checked_cf(cf_dir, [r["id"] for r in rows])
                pretrain(policy, paths, summary["rho_tokens_per_ms"], runtime.cfg, seed)
            runtime.warmup([policy])
            best = validate_policy(runtime, policy, dev_rows)["wall_ms"]
            atomic_json(directory / "dev_initial.json", {"wall_ms": best, "prompts": len(dev_rows)})
            checkpoint(directory / "best.pt", policy, identity=identity, selected_on="dev", epoch=-1)
            epoch = offset = 0
        def save_progress():
            checkpoint(latest, policy, optimizer, identity=identity, epoch=epoch, offset=offset,
                       best=best, sampling_rng=generator.get_state(), torch_rng=torch.get_rng_state())
        save_progress()
        runtime.warmup([policy])
        epochs = 0 if variant == "no_online_rl" else runtime.cfg["train_epochs"]
        while epoch < epochs:
            order = list(range(len(rows)))
            random.Random(f"{seed}:{epoch}").shuffle(order)
            for position in range(offset, len(order)):
                row = rows[order[position]]
                result = runtime.generate(runtime.encode(row["prompt"]), "layered", policy,
                                          sample=True, generator=generator)
                metrics = policy_update(policy, optimizer, result, runtime.cfg["value_coef"])
                atomic_json(directory / "learning_curve" / f"{epoch:03d}_{position:06d}.json", {
                    "id": row["id"], "dataset": row["dataset"], "epoch": epoch, "position": position,
                    **metrics, "wall_ms": result["wall_ms"], "tokens": len(result["tokens"]),
                    "accepted": sum(r["accepted"] for r in result["rounds"]),
                    "nodes": sum(r["nodes"] for r in result["rounds"]),
                    "policy_ms": sum(r["policy_ms"] for r in result["rounds"]),
                    "build_ms": sum(r["build_ms"] for r in result["rounds"])})
                offset = position + 1
                # Checkpoint every complete prompt: crash recovery never reuses an old
                # trajectory with a newly updated policy, and never skips a prompt.
                save_progress()
                print(f"train {variant} seed={seed} epoch={epoch + 1} {offset}/{len(rows)} return={metrics['return']:.4f}", flush=True)
            score = validate_policy(runtime, policy, dev_rows)
            atomic_json(directory / f"dev_epoch_{epoch:03d}.json", score)
            if score["wall_ms"] < best:
                best = score["wall_ms"]
                checkpoint(directory / "best.pt", policy, identity=identity, selected_on="dev", epoch=epoch)
            epoch += 1
            offset = 0
            save_progress()
        atomic_json(directory / "complete.json", {"best_sha256": file_hash(directory / "best.pt"),
            "train_prompts": len(rows), "dev_prompts": len(dev_rows), "epochs": epochs,
            "variant": variant, "seed": seed, "identity": identity})
        return directory / "best.pt"
