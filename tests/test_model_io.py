from pathlib import Path

from non_linear_steering.model_io import preflight_report, resolve_snapshot


def test_resolve_snapshot_uses_ref(tmp_path: Path):
    root = tmp_path / "models--org--name"
    snapshot = root / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    refs = root / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc", encoding="utf-8")
    assert resolve_snapshot(root) == snapshot


def test_preflight_reports_missing_weights(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["Dummy"],"model_type":"dummy"}',
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    ok, messages = preflight_report(tmp_path)
    assert not ok
    assert any("weights: MISSING" in item for item in messages)
