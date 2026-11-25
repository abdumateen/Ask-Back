"""§4.3 per-policy bounds metrics — regression for load_episodes / policy_metrics."""

from pathlib import Path

from askback.metrics import load_episodes, policy_metrics


def _row(policy: str, oracle) -> dict:
    return {"policy": policy,
            "bound_oracle_questions": oracle,
            "bound_no_interaction_pass": oracle is not None}


def test_policy_metrics_groups_by_policy() -> None:
    episodes = [
        _row("P0", 2),
        _row("P0", None),
        _row("P0", 0),
        _row("P1", 5),
        _row("P1", None),
    ]
    report = policy_metrics(episodes)
    assert set(report) == {"P0", "P1"}
    assert report["P0"] == {
        "n_episodes": 3,
        "n_passable": 2,
        "n_episodes_unpassable": 1,
        "mean_bound_oracle_questions": 1.0,
    }
    assert report["P1"] == {
        "n_episodes": 2,
        "n_passable": 1,
        "n_episodes_unpassable": 1,
        "mean_bound_oracle_questions": 5.0,
    }


def test_all_unpassable_yields_none_mean() -> None:
    report = policy_metrics([_row("P2", None), _row("P2", None)])
    assert report["P2"]["n_passable"] == 0
    assert report["P2"]["n_episodes_unpassable"] == 2
    assert report["P2"]["mean_bound_oracle_questions"] is None


def test_load_episodes_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text('{"policy": "P0", "bound_oracle_questions": 1}\n'
                    '{"policy": "P1", "bound_oracle_questions": null}\n',
                    encoding="utf-8")
    rows = load_episodes(path)
    assert [r["policy"] for r in rows] == ["P0", "P1"]
    assert rows[1]["bound_oracle_questions"] is None
