from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class PromptRow(BaseModel):
    prompt: str


class ProbeRow(BaseModel):
    prompt: str
    completion: str
    label: float


def read_prompt_jsonl(path: str | Path) -> list[str]:
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                prompts.append(PromptRow.model_validate(item).prompt)
            else:
                prompts.append(str(item))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def read_probe_jsonl(path: str | Path) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(ProbeRow.model_validate(json.loads(line)))
    if not rows:
        raise ValueError(f"No probe rows found in {path}")
    return rows
