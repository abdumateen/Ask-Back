"""§4.4–4.5, 4.7 eval modules — regression fixtures (A14 grid/API)."""

from pathlib import Path

import numpy as np
import pytest

from askback.eval.diagnostics import diagnostics, episode_blind_share
from askback.eval.frontier import GRID, PRIMARY_V, FALLBACK_V, point_envelope
from askback.eval.stats import (
    holm_bonferroni, quantile, bootstrap_quantile_diff, choose_v,
)


def _row(policy, idx, turns, validity, initial_g, fields):
    return {
        "policy": policy,
        "episode_idx": idx,
        "turns": turns,
        "validity": validity,
        "initial": {"g": initial_g},
        "records": [{"field": f} for f in fields],
    }


def test_point_envelope_quantile_definition() -> None:
    # 20 episodes, all valid; turns sorted 1..20. need = ceil(v · 20).
    turns = list(range(1, 21))
    valid = [True] * 20
    env = point_envelope(turns, valid, 20)
    assert env["0.05"] == 1
    assert env["0.1"] == 2
    assert env["0.3"] == 6
    assert env[str(PRIMARY_V)] == 2
    assert env[str(FALLBACK_V)] == 1
    assert len(env) == len(GRID) == 6
    assert env["0.15"] == 3


def test_quantile_unattainable_is_none() -> None:
    turns = [1, 2, 3]
    valid = [True, True, True]
    assert quantile(turns, valid, 0.6, total=5) == 3   # need ceil(3)=3
    assert quantile(turns, valid, 0.9, total=5) is None  # need ceil(4.5)=5 > 3


def _ab(base_turns, p3_turns, n=40):
    Ta = {i: (base_turns, True) for i in range(n)}
    Tb = {i: (p3_turns, True) for i in range(n)}
    return Ta, Tb, list(range(n))


def test_bootstrap_quantile_diff_one_sided_lower() -> None:
    rng = np.random.default_rng(0)
    Ta, Tb, common = _ab(5, 3)
    res = bootstrap_quantile_diff(Ta, Tb, common, 0.5,
                                  n_resamples=1000, rng=rng)
    assert res["observed_quantile_diff"] == 2
    assert res["p_one_sided"] < 0.05


def test_bootstrap_quantile_diff_null_effect_p_not_small() -> None:
    rng = np.random.default_rng(1)
    Ta, Tb, common = _ab(5, 5)
    res = bootstrap_quantile_diff(Ta, Tb, common, 0.5,
                                  n_resamples=1000, rng=rng)
    assert res["observed_quantile_diff"] == 0
    assert res["p_one_sided"] > 0.05


def test_holm_monotone_and_floor() -> None:
    adj = holm_bonferroni([0.01, 0.04, 0.05])
    # k=3: running max of (3·0.01, 2·0.04, 1·0.05) = (0.03, 0.08, 0.08)
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.08)
    assert adj[2] == pytest.approx(0.08)


def _make_choose_v_data(s0_valid):
    data = {}
    for pol in ("P0", "P1", "P2", "P3"):
        nv = s0_valid.get(pol, 100)
        data[pol] = {"S0:None": {i: (3, i < nv) for i in range(100)}}
    return data


def test_choose_v_primary() -> None:
    data = _make_choose_v_data({})
    v, reason = choose_v(data)
    assert v == 0.10


def test_choose_v_fallback() -> None:
    # P0 has 8/100 passing: T(0.10) needs 10 -> undefined; T(0.05) needs 5 -> ok.
    data = _make_choose_v_data({"P0": 8})
    v, reason = choose_v(data)
    assert v == 0.05


def test_choose_v_vacuous() -> None:
    # P0 has 4/100 passing: even T(0.05) needs 5 -> undefined.
    data = _make_choose_v_data({"P0": 4})
    v, reason = choose_v(data)
    assert v is None


def test_episode_blind_share() -> None:
    row = _row("P0", 0, 3, True, [1, 2], [0, 1, 2])
    share, _, _ = episode_blind_share(row)
    assert share == pytest.approx(2 / 3)


def test_diagnostics_no_data_ok(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"run_id": "t", "policies": ["P0"], "master_seed": 0}',
        encoding="utf-8")
    (tmp_path / "episodes.jsonl").write_text("", encoding="utf-8")
    report = diagnostics(tmp_path)
    assert report["overall_total_asks"] == 0
    assert report["overall_blind_spot_targeting_rate"] is None
