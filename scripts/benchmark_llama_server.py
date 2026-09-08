#!/usr/bin/env python3
"""Benchmark a persistent llama-server on this project's fixed prompt JSONL."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def request_completion(url: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict, float]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    rows = load_rows(args.prompts, args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Warm up kernels and server-side buffers without contaminating the result file.
    request_completion(args.url, "Warmup", min(args.max_new_tokens, 16), args.timeout)

    with args.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    result, client_ms = request_completion(
                        args.url, row["prompt"], args.max_new_tokens, args.timeout
                    )
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    last_error = exc
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            else:  # pragma: no cover
                raise RuntimeError(last_error)

            choice = result.get("choices", [{}])[0]
            timings = result.get("timings", {})
            record = {
                "index": index,
                "dataset": row.get("dataset"),
                "source_id": row.get("source_id"),
                "prompt": row["prompt"],
                "implementation": args.implementation,
                "content": choice.get("message", {}).get("content", ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": result.get("usage", {}),
                "timings": timings,
                "client_ms": client_ms,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "index": index,
                        "implementation": args.implementation,
                        "predicted_per_second": timings.get("predicted_per_second"),
                        "draft_n": timings.get("draft_n"),
                        "draft_n_accepted": timings.get("draft_n_accepted"),
                        "client_ms": client_ms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
