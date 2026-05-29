#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask external LLM reviewers for structured code-review suggestions.

Reads all secrets from environment variables and never prints API keys.
Doubao/Ark uses the Responses API; DeepSeek uses chat completions.
"""

import argparse
import json
import os
import subprocess
import sys
import ssl
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


REVIEW_PROMPT = """你是德州扑克强化学习项目的代码审查 agent。
请重点审查：
1. 训练正确性：PPO/DQN更新、奖励、轨迹、动作合法性。
2. 德州扑克规则：盲注、行动顺序、摊牌、平局、all-in。
3. 性能瓶颈：CPU环境采样、GPU利用率、批量推理。
4. 可验证性：最小测试、指标、实验设计。

输出 JSON，格式：
{
  "summary": "...",
  "findings": [
    {"severity": "high|medium|low", "file": "...", "line_hint": "...", "issue": "...", "suggestion": "..."}
  ],
  "experiments": ["..."]
}
不要输出 Markdown，不要泛泛而谈。
"""


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def collect_context(max_diff_chars: int, max_tree_chars: int) -> str:
    branch = run(["git", "status", "--short", "--branch"])
    diff = run(["git", "diff", "--", "poker_ai", "train.py", "evaluate.py", "tools"])
    tracked = run(["git", "ls-files"])
    tree = "\n".join(
        line for line in tracked.splitlines()
        if line.endswith(".py") or line.endswith(".md") or line == "requirements.txt"
    )
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n...[diff truncated]..."
    if len(tree) > max_tree_chars:
        tree = tree[:max_tree_chars] + "\n...[tree truncated]..."
    return f"BRANCH_STATUS:\n{branch}\n\nPROJECT_TREE:\n{tree}\n\nCURRENT_DIFF:\n{diff}\n"


def _ssl_context():
    if os.getenv("AGENT_REVIEW_INSECURE_SSL") == "1":
        return ssl._create_unverified_context()
    return None


def _post_json(url: str, api_key: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_jsonish(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.removeprefix("json").strip()
        try:
            return json.loads(stripped)
        except Exception:
            return {"raw": content}


def doubao_responses(name: str, base_url: str, api_key: str, model: str, context: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": REVIEW_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": context}],
            },
        ],
        "temperature": 0.2,
    }
    try:
        data = _post_json(url, api_key, payload, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"reviewer": name, "ok": False, "error": f"HTTP {exc.code}: {body[:800]}"}
    except Exception as exc:
        return {"reviewer": name, "ok": False, "error": str(exc)}

    content = data.get("output_text", "")
    if not content:
        chunks = []
        for item in data.get("output", []):
            for part in item.get("content", []):
                text = part.get("text") or part.get("content") or ""
                if text:
                    chunks.append(text)
        content = "\n".join(chunks)
    return {"reviewer": name, "ok": True, "model": model, "result": _parse_jsonish(content)}


def chat_completion(name: str, base_url: str, api_key: str, model: str, context: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": context},
        ],
        "temperature": 0.2,
    }
    try:
        data = _post_json(url, api_key, payload, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"reviewer": name, "ok": False, "error": f"HTTP {exc.code}: {body[:800]}"}
    except Exception as exc:
        return {"reviewer": name, "ok": False, "error": str(exc)}

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"reviewer": name, "ok": True, "model": model, "result": _parse_jsonish(content)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Doubao and DeepSeek code-review agents.")
    parser.add_argument("--out", default="agent_reviews", help="Output directory")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-diff-chars", type=int, default=50000)
    parser.add_argument("--max-tree-chars", type=int, default=12000)
    args = parser.parse_args()

    reviewers = [
        {
            "name": "doubao",
            "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            "api_key": os.getenv("ARK_API_KEY") or os.getenv("DOUBAO_API_KEY", ""),
            "model": os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260428"),
            "api": "responses",
        },
        {
            "name": "deepseek",
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            "api": "chat",
        },
    ]

    context = collect_context(args.max_diff_chars, args.max_tree_chars)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"review_{timestamp}.json"

    results = []
    for reviewer in reviewers:
        if not reviewer["api_key"]:
            results.append({"reviewer": reviewer["name"], "ok": False, "error": "missing API key env var"})
            continue
        fn = doubao_responses if reviewer["api"] == "responses" else chat_completion
        results.append(fn(reviewer["name"], reviewer["base_url"], reviewer["api_key"], reviewer["model"], context, args.timeout))

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))
    failed = [item for item in results if not item.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
