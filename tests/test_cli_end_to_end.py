"""End-to-end CLI run against a tiny locally-saved checkpoint

This exercises the real paths the documented commands take: snapshot resolution,
preflight, tokenizer/model loading, probe training, and steering training
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from non_linear_steering.cli import main
from non_linear_steering.probe import load_probe

VOCAB = {"<unk>": 0, "<pad>": 1, "<eos>": 2, "a": 3, "b": 4, "c": 5}


@pytest.fixture(scope="module")
def local_model(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-model")
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=VOCAB["<eos>"],
        pad_token_id=VOCAB["<pad>"],
    )
    LlamaForCausalLM(config).save_pretrained(directory)

    backend = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    ).save_pretrained(directory)
    return directory


def write_probe_dataset(path: Path) -> None:
    rows = [
        {"prompt": "a b", "completion": "c a", "label": 1},
        {"prompt": "a b", "completion": "b b", "label": 0},
        {"prompt": "b c", "completion": "a c", "label": 1},
        {"prompt": "b c", "completion": "c c", "label": 0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def write_prompts(path: Path) -> None:
    rows = [{"prompt": "a b"}, {"prompt": "b c a"}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_preflight_accepts_a_complete_checkpoint(local_model, capsys):
    assert main(["preflight", "--model", str(local_model)]) == 0
    assert "status: OK" in capsys.readouterr().out


def test_probe_then_steering(local_model, tmp_path, capsys):
    dataset = tmp_path / "probe_train.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    probe_path = tmp_path / "checkpoints" / "probe.pt"
    steering_path = tmp_path / "checkpoints" / "steering.pt"
    write_probe_dataset(dataset)
    write_prompts(prompts)
    common = ["--model", str(local_model), "--dtype", "float32", "--device", "cpu"]

    assert (
        main(
            [
                "train-probe",
                *common,
                "--dataset",
                str(dataset),
                "--probe-layer",
                "2",
                "--probe-width",
                "16",
                "--probe-layers",
                "1",
                "--probe-heads",
                "2",
                "--batch-size",
                "2",
                "--epochs",
                "2",
                "--output",
                str(probe_path),
            ]
        )
        == 0
    )
    assert probe_path.exists()
    probe, probe_layer = load_probe(probe_path)
    assert probe_layer == 2
    assert probe.config.hidden_size == 32

    capsys.readouterr()
    assert (
        main(
            [
                "train-steering",
                *common,
                "--prompts",
                str(prompts),
                "--probe",
                str(probe_path),
                "--inject-layer",
                "1",
                "--probe-layer",
                "2",
                "--horizon",
                "4",
                "--rank",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "3",
                "--lr",
                "1e-2",
                "--output",
                str(steering_path),
            ]
        )
        == 0
    )
    records = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert len(records) == 3
    for record in records:
        for key in ("loss", "score", "kl", "beta", "grad_norm"):
            assert record[key] == record[key], f"{key} is NaN"
        assert record["kl"] >= 0.0
        assert record["beta"] >= 0.0
    assert any(record["grad_norm"] > 0.0 for record in records)

    checkpoint = torch.load(steering_path, map_location="cpu")
    assert checkpoint["inject_layer"] == 1
    assert checkpoint["probe_layer"] == 2
    assert checkpoint["hidden_size"] == 32
    assert checkpoint["rank"] == 4
    assert checkpoint["steer_position"] == "prediction-state"
    assert checkpoint["state_dict"]["deltas.1.up.weight"].abs().sum() > 0.0


def test_steering_rejects_a_probe_from_another_layer(local_model, tmp_path):
    dataset = tmp_path / "probe_train.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    probe_path = tmp_path / "probe.pt"
    write_probe_dataset(dataset)
    write_prompts(prompts)
    common = ["--model", str(local_model), "--dtype", "float32", "--device", "cpu"]
    main(
        [
            "train-probe",
            *common,
            "--dataset",
            str(dataset),
            "--probe-layer",
            "2",
            "--probe-width",
            "16",
            "--probe-layers",
            "1",
            "--probe-heads",
            "2",
            "--output",
            str(probe_path),
        ]
    )
    with pytest.raises(ValueError, match="trained at layer 2"):
        main(
            [
                "train-steering",
                *common,
                "--prompts",
                str(prompts),
                "--probe",
                str(probe_path),
                "--inject-layer",
                "1",
                "--probe-layer",
                "3",
                "--steps",
                "1",
                "--output",
                str(tmp_path / "steering.pt"),
            ]
        )


def test_steering_rejects_layers_beyond_the_model(local_model, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    write_prompts(prompts)
    with pytest.raises(ValueError, match="out of range"):
        main(
            [
                "train-steering",
                "--model",
                str(local_model),
                "--dtype",
                "float32",
                "--device",
                "cpu",
                "--prompts",
                str(prompts),
                "--probe",
                str(tmp_path / "missing.pt"),
                "--inject-layer",
                "1",
                "--probe-layer",
                "9",
                "--steps",
                "1",
                "--output",
                str(tmp_path / "steering.pt"),
            ]
        )
