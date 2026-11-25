"""§14 CLI wiring: config parity, resume, 32-point replay (T-10 at the CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askback.runner.cli import (
    _read_jsonl,
    config_hash,
    load_config,
    replay_study,
    run_study,
    stop_points,
    validate_config,
)

CORE = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"


def _cfg(n_episodes: int = 2, policies=("P0", "P3")) -> dict:
    cfg = load_config(CORE)
    cfg["policies"] = list(policies)
    cfg["n_episodes"] = n_episodes
    return cfg


def test_core_yaml_matches_code() -> None:
    validate_config(load_config(CORE))


def test_validate_refuses_voi_drift() -> None:
    cfg = _cfg()
    cfg["voi"]["exact_max_states"] = 20000  # §14.1's pre-amendment sample
    with pytest.raises(ValueError, match="exact_max_states"):
        validate_config(cfg)


def test_validate_refuses_p2() -> None:
    cfg = _cfg()
    cfg["extractor"]["type"] = "p2"
    with pytest.raises(ValueError, match="§5"):
        validate_config(cfg)


def test_stop_points_cover_the_a2_grid() -> None:
    points = stop_points(_cfg())
    assert [p[0] for p in points] == ["S0"] + ["S1"] * 5 + ["S2"] * 2
    assert [p[3] for p in points] == [None, 0.005, 0.01, 0.02, 0.05, 0.10, 0.7, 1.0]
    assert len({(r, p) for r, _, _, p in points}) == 8


def test_run_and_replay_write_the_declared_rows(tmp_path: Path) -> None:
    cfg = _cfg()
    run_study(cfg, tmp_path)
    replay_study(tmp_path)

    episodes = _read_jsonl(tmp_path / "episodes.jsonl")
    points = _read_jsonl(tmp_path / "points.jsonl")
    assert len(episodes) == 2 * 2
    assert len(points) == 2 * 8

    row = episodes[0]
    assert set(row["initial"]) == {"u", "g", "delta_p", "validator_pass"}
    assert len(row["trajectory_hash"]) == 64
    assert set(row["seed"]) == {"entropy", "spawn_key"}
    assert row["config_hash"] == config_hash(cfg)
    for key in ("redundant_question_rate", "wasted_turn_rate", "stopping_quality"):
        assert key in points[0]
    assert set(points[0]["stopping_quality"]) == {"too_early", "too_late"}


def test_points_unique_key(tmp_path: Path) -> None:
    cfg = _cfg(policies=("P0", "P1", "P2", "P3"))
    run_study(cfg, tmp_path)
    replay_study(tmp_path)
    points = _read_jsonl(tmp_path / "points.jsonl")
    keys = [(p["policy"], p["stop_rule"], p["param"]) for p in points]
    assert len(keys) == len(set(keys)) == 32


def test_points_unique_key_multi_persona(tmp_path: Path) -> None:
    cfg = _cfg(policies=("P0", "P1", "P2", "P3"))
    cfg["personas"] = ["realistic", "cooperative", "impatient_expert"]
    run_study(cfg, tmp_path)
    replay_study(tmp_path)
    points = _read_jsonl(tmp_path / "points.jsonl")
    keys = [
        (p["policy"], p["persona"], p["density"], p["stop_rule"], p["param"])
        for p in points
    ]
    assert len(keys) == len(set(keys)) == 96


def test_resume_is_byte_identical(tmp_path: Path) -> None:
    cfg = _cfg()
    run_study(cfg, tmp_path)
    first = (tmp_path / "episodes.jsonl").read_bytes()
    run_study(cfg, tmp_path)  # every episode already seen
    assert (tmp_path / "episodes.jsonl").read_bytes() == first


def test_two_runs_match_byte_for_byte(tmp_path: Path) -> None:
    """A8 / T-10: same config, distinct dirs, identical points.jsonl."""
    cfg = _cfg()
    a, b = tmp_path / "a", tmp_path / "b"
    run_study(cfg, a)
    run_study(cfg, b)
    replay_study(a)
    replay_study(b)
    assert json.loads((a / "config.json").read_text()) == json.loads(
        (b / "config.json").read_text()
    )
    assert (a / "points.jsonl").read_bytes() == (b / "points.jsonl").read_bytes()
    assert (a / "episodes.jsonl").read_bytes() == (b / "episodes.jsonl").read_bytes()