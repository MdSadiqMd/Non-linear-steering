from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from pydantic import BaseModel, Field

from .config import (
    DEFAULT_MODEL,
    STEER_POSITIONS,
    HookSpec,
    ProbeConfig,
    SteerPosition,
    TrainSpec,
)
from .model_io import preflight_report


class ModelArgs(BaseModel):
    model: Path = DEFAULT_MODEL
    dtype: str = "auto"
    device: str = "auto"
    seed: int = 0


class PreflightArgs(BaseModel):
    model: Path = DEFAULT_MODEL


class ProbeArgs(ModelArgs):
    dataset: Path
    output: Path = Path("checkpoints/probe.pt")
    probe_layer: int = Field(ge=0)
    probe_width: int = Field(default=512, gt=0)
    probe_layers: int = Field(default=2, gt=0)
    probe_heads: int = Field(default=8, gt=0)
    batch_size: int = Field(default=1, gt=0)
    epochs: int = Field(default=1, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-3, ge=0)


class SteeringArgs(ModelArgs):
    prompts: Path
    probe: Path
    output: Path = Path("checkpoints/steering.pt")
    inject_layer: int = Field(ge=0)
    probe_layer: int = Field(ge=0)
    steer_position: SteerPosition = "prediction-state"
    horizon: int = Field(default=64, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    epsilon: float = Field(default=0.1, ge=0)
    dual_lr: float = Field(default=0.05, ge=0)
    radius: float = Field(default=0.5, gt=0)
    rank: int = Field(default=32, gt=0)
    batch_size: int = Field(default=1, gt=0)
    steps: int = Field(default=100, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-3, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    beta: float = 0.0
    baseline: float = 0.0
    baseline_decay: float = Field(default=0.95, ge=0, lt=1)
    log_every: int = Field(default=1, gt=0)


def _resolve_dtype(name: str):
    import torch

    return torch.float16 if name == "auto" else getattr(torch, name)


def _resolve_device(name: str) -> str:
    import torch

    return name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_preflight(namespace: argparse.Namespace) -> int:
    args = PreflightArgs.model_validate(vars(namespace))
    ok, messages = preflight_report(args.model)
    print("\n".join(messages))
    return 0 if ok else 2


def cmd_train_probe(namespace: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from .data import read_probe_jsonl
    from .hooks import HookedModel, load_hooked_model
    from .probe import CausalProbe, save_probe
    from .trajectory import padding_side

    args = ProbeArgs.model_validate(vars(namespace))
    torch.manual_seed(args.seed)

    model = load_hooked_model(
        args.model, dtype=_resolve_dtype(args.dtype), device=_resolve_device(args.device)
    )
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = str(model.cfg.device)

    hooked = HookedModel(model=model, inject_layers=[], probe_layer=args.probe_layer)
    rows = read_probe_jsonl(args.dataset)
    hidden_size = model.cfg.d_model
    probe = CausalProbe(
        ProbeConfig(
            hidden_size=hidden_size,
            probe_width=args.probe_width,
            layers=args.probe_layers,
            heads=args.probe_heads,
        )
    ).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eos = tokenizer.eos_token or ""
    shuffle = torch.Generator().manual_seed(args.seed)

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        order = torch.randperm(len(rows), generator=shuffle)
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"probe epoch {epoch + 1}"):
            chunk = [rows[int(index)] for index in order[start : start + args.batch_size]]
            texts = [row.prompt + row.completion + eos for row in chunk]
            labels = torch.tensor([row.label for row in chunk], device=device)
            with padding_side(tokenizer, "right"):
                encoded = tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=True
                )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            terminal = attention_mask.long().sum(dim=-1) - 1
            with torch.no_grad(), hooked.session(capture=True) as state:
                hooked.forward(input_ids, attention_mask=attention_mask)
                stream = state.captured
            if stream is None:
                raise RuntimeError("Probe stream was not captured")
            logits = probe(stream.detach().to(device), attention_mask.to(device))
            selected = logits[torch.arange(logits.shape[0], device=device), terminal]
            loss = F.binary_cross_entropy_with_logits(selected, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(chunk)
            count += len(chunk)
        print(json.dumps({"epoch": epoch + 1, "loss": total / max(count, 1)}))

    save_probe(args.output, probe.cpu(), probe_layer=args.probe_layer)
    print(f"saved_probe: {args.output}")
    return 0


def cmd_train_steering(namespace: argparse.Namespace) -> int:
    import torch
    from tqdm import tqdm

    from .data import read_prompt_jsonl
    from .hooks import load_hooked_model
    from .objective import constrained_loss, replay_and_score
    from .probe import load_probe
    from .steering import LayerwiseSteering
    from .trajectory import sample_rollout

    args = SteeringArgs.model_validate(vars(namespace))
    torch.manual_seed(args.seed)

    spec = TrainSpec(
        horizon=args.horizon,
        temperature=args.temperature,
        epsilon=args.epsilon,
        dual_lr=args.dual_lr,
        radius=args.radius,
        rank=args.rank,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        baseline_decay=args.baseline_decay,
    )
    hook = HookSpec(
        inject_layer=args.inject_layer,
        probe_layer=args.probe_layer,
        steer_position=args.steer_position,
    )

    model = load_hooked_model(
        args.model, dtype=_resolve_dtype(args.dtype), device=_resolve_device(args.device)
    )
    device = str(model.cfg.device)
    hidden_size = model.cfg.d_model
    hook.validate_for_model(num_layers=model.cfg.n_layers)
    if not hook.activation_channel_open:
        print(
            json.dumps(
                {
                    "warning": "inject_layer is downstream of probe_layer, so the probe "
                    "score has no direct gradient in the steering parameters; only the "
                    "score-function term will train"
                }
            )
        )

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe, saved_probe_layer = load_probe(args.probe, map_location=device)
    if saved_probe_layer != args.probe_layer:
        raise ValueError(f"Probe was trained at layer {saved_probe_layer}, not {args.probe_layer}")
    if probe.config.hidden_size != hidden_size:
        raise ValueError(
            f"Probe expects hidden_size={probe.config.hidden_size}, model has {hidden_size}"
        )
    probe = probe.to(device).eval()

    steering = LayerwiseSteering(hidden_size, [args.inject_layer], spec.rank, spec.radius).to(
        device
    )
    opt = torch.optim.AdamW(steering.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prompts = read_prompt_jsonl(args.prompts)
    prompt_cycle = itertools.cycle(prompts)
    beta = args.beta
    baseline = args.baseline

    for step in tqdm(range(1, args.steps + 1), desc="steering"):
        batch_prompts = [next(prompt_cycle) for _ in range(args.batch_size)]
        rollout = sample_rollout(
            model,
            tokenizer,
            steering,
            batch_prompts,
            inject_layer=args.inject_layer,
            horizon=args.horizon,
            temperature=args.temperature,
            steer_position=args.steer_position,
        )
        values = replay_and_score(
            model,
            probe,
            steering,
            rollout,
            inject_layer=args.inject_layer,
            probe_layer=args.probe_layer,
            temperature=args.temperature,
        )
        loss = constrained_loss(values, beta=beta, baseline=baseline)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(steering.parameters(), args.grad_clip)
        opt.step()

        mean_kl = values.seq_kl.detach().mean().item()
        batch_return = (values.score.detach() - beta * values.seq_kl.detach()).mean().item()
        if step % args.log_every == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "score": values.score.detach().mean().item(),
                        "kl": mean_kl,
                        "beta": beta,
                        "baseline": baseline,
                        "truncated_rate": rollout.truncated.float().mean().item(),
                        "grad_norm": float(grad_norm),
                    }
                )
            )
        baseline = spec.baseline_decay * baseline + (1 - spec.baseline_decay) * batch_return
        beta = max(0.0, beta + spec.dual_lr * (mean_kl - spec.epsilon))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": steering.state_dict(),
            "hidden_size": hidden_size,
            "inject_layer": args.inject_layer,
            "probe_layer": args.probe_layer,
            "steer_position": args.steer_position,
            "rank": spec.rank,
            "radius": spec.radius,
            "temperature": spec.temperature,
            "epsilon": spec.epsilon,
            "beta": beta,
            "baseline": baseline,
        },
        args.output,
    )
    print(f"saved_steering: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nls")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--model", default=str(DEFAULT_MODEL))
    preflight.set_defaults(func=cmd_preflight)

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--model", default=str(DEFAULT_MODEL))
    common_model.add_argument("--dtype", default="auto")
    common_model.add_argument("--device", default="auto")
    common_model.add_argument("--seed", type=int, default=0)

    probe = sub.add_parser("train-probe", parents=[common_model])
    probe.add_argument("--dataset", required=True)
    probe.add_argument("--output", default="checkpoints/probe.pt")
    probe.add_argument("--probe-layer", type=int, required=True)
    probe.add_argument("--probe-width", type=int, default=512)
    probe.add_argument("--probe-layers", type=int, default=2)
    probe.add_argument("--probe-heads", type=int, default=8)
    probe.add_argument("--batch-size", type=int, default=1)
    probe.add_argument("--epochs", type=int, default=1)
    probe.add_argument("--lr", type=float, default=1e-4)
    probe.add_argument("--weight-decay", type=float, default=1e-3)
    probe.set_defaults(func=cmd_train_probe)

    steering = sub.add_parser("train-steering", parents=[common_model])
    steering.add_argument("--prompts", required=True)
    steering.add_argument("--probe", required=True)
    steering.add_argument("--output", default="checkpoints/steering.pt")
    steering.add_argument("--inject-layer", type=int, required=True)
    steering.add_argument("--probe-layer", type=int, required=True)
    steering.add_argument(
        "--steer-position", default="prediction-state", choices=list(STEER_POSITIONS)
    )
    steering.add_argument("--horizon", type=int, default=64)
    steering.add_argument("--temperature", type=float, default=1.0)
    steering.add_argument("--epsilon", type=float, default=0.1)
    steering.add_argument("--dual-lr", type=float, default=0.05)
    steering.add_argument("--radius", type=float, default=0.5)
    steering.add_argument("--rank", type=int, default=32)
    steering.add_argument("--batch-size", type=int, default=1)
    steering.add_argument("--steps", type=int, default=100)
    steering.add_argument("--lr", type=float, default=1e-4)
    steering.add_argument("--weight-decay", type=float, default=1e-3)
    steering.add_argument("--grad-clip", type=float, default=1.0)
    steering.add_argument("--beta", type=float, default=0.0)
    steering.add_argument("--baseline", type=float, default=0.0)
    steering.add_argument("--baseline-decay", type=float, default=0.95)
    steering.add_argument("--log-every", type=int, default=1)
    steering.set_defaults(func=cmd_train_steering)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
