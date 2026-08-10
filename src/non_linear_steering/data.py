from __future__ import annotations

import json
from pathlib import Path


def read_prompt_jsonl(path: str | Path) -> list[str]:
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(item["prompt"] if isinstance(item, dict) else str(item))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def read_probe_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if "prompt" not in item or "completion" not in item or "label" not in item:
                raise ValueError("Probe rows must contain prompt, completion, and label fields")
            rows.append(item)
    if not rows:
        raise ValueError(f"No probe rows found in {path}")
    return rows
