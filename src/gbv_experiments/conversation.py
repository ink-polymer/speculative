from __future__ import annotations

from collections import defaultdict

from .common import canonical, digest, prompt_seed
from .data import user_turns


def encode_messages(tokenizer, messages, model, device):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                     enable_thinking=bool(model.get("enable_thinking", False)))
    return tokenizer(formatted, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def aggregate_turns(turns):
    if not turns:
        raise ValueError("No generated turns")
    if len(turns) == 1:
        return {**turns[0], "turn_count": 1}
    result = {}
    summed = ("generated_tokens", "decode_tokens", "prefill_ms", "decode_ms", "e2e_ms",
              "target_forward_calls", "draft_forward_calls", "target_tokens_processed", "input_tokens")
    for name in summed:
        result[name] = sum(turn[name] for turn in turns)
    result["generated_token_ids"] = [token for turn in turns for token in turn["generated_token_ids"]]
    result["rounds"] = [{**item, "turn_index": i} for i, turn in enumerate(turns, 1) for item in turn["rounds"]]
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [turn[field] for turn in turns if turn[field] is not None]
        result[field] = max(values) if values else None
    stages = {"host_ms": defaultdict(float), "cuda_event_ms": defaultdict(float)}
    for turn in turns:
        for kind, values in turn["stages"].items():
            for name, elapsed in values.items():
                stages[kind][name] += elapsed
    result["stages"] = {k: dict(v) for k, v in stages.items()}
    result["decode_tokens_per_second"] = 1000 * result["decode_tokens"] / result["decode_ms"] if result["decode_tokens"] else None
    result["e2e_tokens_per_second"] = 1000 * result["generated_tokens"] / result["e2e_ms"]
    result["finish_reason"] = "length" if any(t["finish_reason"] == "length" for t in turns) else "eos"
    result["text"] = turns[0]["text"] if len(turns) == 1 else canonical([t["text"] for t in turns])
    result["turn_count"] = len(turns)
    result["turn_results"] = turns
    return result


def generate_conversation(engine, tokenizer, row, variant, max_new_tokens, stop_ids, seed, model, profile=False):
    messages, results = [], []
    for index, prompt in enumerate(user_turns(row)):
        messages.append({"role": "user", "content": prompt})
        ids = encode_messages(tokenizer, messages, model, engine.device)
        turn_seed = seed if index == 0 else prompt_seed(seed, row["dataset"], f"{row['source_id']}:turn:{index + 1}")
        result = engine.generate(ids, variant, max_new_tokens, stop_ids, seed=turn_seed, profile=profile)
        result.update({"turn_index": index + 1, "sampling_seed": turn_seed,
                       "messages_sha256": digest(messages), "input_ids_sha256": digest(ids[0].tolist()),
                       "input_tokens": ids.shape[1]})
        result["text"] = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True)
        results.append(result)
        # Each method uses its own actual first answer for the second turn.
        messages.append({"role": "assistant", "content": result["text"]})
    return aggregate_turns(results)
