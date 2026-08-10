from __future__ import annotations

import json
from pathlib import Path


def _is_hf_model_name(model_path: str | Path) -> bool:
    """Check if model_path looks like a HuggingFace model name (org/model or model)"""
    path = Path(model_path)
    if path.exists():
        return False
    name = str(model_path)
    return "/" in name or not any(c in name for c in ("\\", "~"))


def resolve_snapshot(model_path: str | Path) -> Path:
    model_path = Path(model_path).expanduser()
    if (model_path / "snapshots").is_dir():
        ref = model_path / "refs" / "main"
        if ref.exists():
            revision = ref.read_text(encoding="utf-8").strip()
            snapshot = model_path / "snapshots" / revision
            if snapshot.exists():
                return snapshot
        snapshots = sorted((model_path / "snapshots").iterdir())
        if snapshots:
            return snapshots[-1]
    return model_path


def read_config(model_path: str | Path) -> dict:
    snapshot = resolve_snapshot(model_path)
    config_path = snapshot / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def weight_files(model_path: str | Path) -> list[Path]:
    snapshot = resolve_snapshot(model_path)
    patterns = ["*.safetensors", "*.bin", "*.pt", "*.gguf"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(snapshot.glob(pattern))
    return sorted(files)


def preflight_report(model_path: str | Path) -> tuple[bool, list[str]]:
    if _is_hf_model_name(model_path):
        return True, [f"model_path: {model_path}", "status: HuggingFace model name (will download)"]
    snapshot = resolve_snapshot(model_path)
    messages = [f"model_path: {Path(model_path).expanduser()}", f"snapshot: {snapshot}"]
    try:
        cfg = read_config(model_path)
    except FileNotFoundError as exc:
        messages.append(f"FAIL: {exc}")
        return False, messages

    messages.append(f"architecture: {cfg.get('architectures', ['unknown'])[0]}")
    messages.append(f"model_type: {cfg.get('model_type', 'unknown')}")
    messages.append(f"hidden_size: {cfg.get('hidden_size', 'unknown')}")
    messages.append(f"num_hidden_layers: {cfg.get('num_hidden_layers', 'unknown')}")
    messages.append(f"eos_token_id: {cfg.get('eos_token_id', 'unknown')}")
    messages.append(f"pad_token_id: {cfg.get('pad_token_id', 'unknown')}")

    tokenizer_ok = (snapshot / "tokenizer.json").exists()
    messages.append(f"tokenizer.json: {'OK' if tokenizer_ok else 'MISSING'}")

    weights = weight_files(model_path)
    if weights:
        total = sum(path.stat().st_size for path in weights if path.exists())
        messages.append(f"weights: {len(weights)} files, {total / 1e9:.2f} GB")
    else:
        messages.append("weights: MISSING (*.safetensors/*.bin/*.pt/*.gguf not found)")

    ok = tokenizer_ok and bool(weights)
    messages.append("status: OK" if ok else "status: NOT TRAINABLE YET")
    return ok, messages


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    if _is_hf_model_name(model_path):
        return AutoTokenizer.from_pretrained(str(model_path))
    snapshot = resolve_snapshot(model_path)
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def load_causal_lm(model_path: str | Path, *, dtype: str = "auto", device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    if _is_hf_model_name(model_path):
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    snapshot = resolve_snapshot(model_path)
    return AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
