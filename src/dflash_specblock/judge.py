"""MT-Bench 与 Alpaca 的 judge 接口。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge 返回中没有 JSON 对象")
    return json.loads(text[start : end + 1])


@dataclass(slots=True)
class OpenAIJudgeClient:
    model: str
    api_key: str
    base_url: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenAIJudgeClient | None":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("JUDGE_API_KEY")
        if not api_key:
            return None
        value = model or os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
        if not value:
            return None
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("JUDGE_BASE_URL") or "https://api.openai.com/v1"
        return cls(model=value, api_key=api_key, base_url=base_url.rstrip("/"))

    def _chat(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("judge 返回内容不是字符串")
        return _extract_json_object(content)

    def judge_mt_bench(self, prompt: str, response: str, category: str | None) -> dict[str, Any]:
        system_prompt = (
            "You are an impartial LLM judge for MT-Bench style evaluation. "
            "Score the assistant answer from 1 to 10 for helpfulness, correctness, completeness, and relevance. "
            "Return strict JSON with keys score and rationale."
        )
        user_prompt = (
            f"Category: {category or 'unknown'}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Assistant Answer:\n{response}\n"
        )
        result = self._chat(system_prompt, user_prompt)
        score = float(result["score"])
        return {"judge_score": score, "rationale": str(result.get("rationale", ""))}

    def judge_alpaca(self, prompt: str, response: str, reference: str) -> dict[str, Any]:
        system_prompt = (
            "You compare a candidate answer against a reference answer for Alpaca-style instruction following. "
            "Return strict JSON with keys verdict and rationale. "
            "verdict must be one of better, tie, worse."
        )
        user_prompt = (
            f"Instruction:\n{prompt}\n\n"
            f"Candidate Answer:\n{response}\n\n"
            f"Reference Answer:\n{reference}\n"
        )
        result = self._chat(system_prompt, user_prompt)
        verdict = str(result["verdict"]).strip().lower()
        if verdict not in {"better", "tie", "worse"}:
            raise ValueError(f"非法 Alpaca judge verdict: {verdict}")
        win_rate = 1.0 if verdict == "better" else 0.5 if verdict == "tie" else 0.0
        return {
            "verdict": verdict,
            "win_rate": win_rate,
            "rationale": str(result.get("rationale", "")),
        }
