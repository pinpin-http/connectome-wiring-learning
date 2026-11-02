"""Reading functions of the §12 probe (docs/RL_STUDY.md §12) on hand-built episodes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gates"))

import frozen_play_probe as fp  # noqa: E402


def _row(label, jump, end="death", level="Level5-1", best=300.0, rand=300.0):
    return {"learner": "x", "level": level, "delay": 0, "best_x": best, "random_best_x": rand, "end": end,
            "log": {"label": [bool(v) for v in label], "jump": [bool(v) for v in jump],
                    "p": [float(v) for v in jump], "act": [0] * len(label), "x": [0.0] * len(label)}}


def test_timing_lag_miss_press_and_failure_context():
    t = fp.timing([_row([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0])])
    assert t["onsets"] == 2 and t["missed"] == 1 and t["miss_rate"] == 0.5
    assert t["median_lag"] == 1.0
    assert t["press_length_median"] == 2.0 and t["press_length_share_1"] == 0.0
    assert t["jump_while_label_off"] == 0.0
    assert t["failures"] == {"missed": 0, "pressed": 1, "unlabelled": 0}


def test_timing_expert_is_perfect_and_failure_kinds():
    lab = [0, 1, 1, 0, 1, 0]
    t = fp.timing([_row(lab, lab, end="flag")])
    assert t["missed"] == 0 and t["median_lag"] == 0.0 and t["jump_while_label_off"] == 0.0
    assert t["failures"] == {"missed": 0, "pressed": 0, "unlabelled": 0}
    t = fp.timing([_row([0, 1, 1, 0], [0, 0, 0, 1], end="stall"), _row([0, 0, 0], [1, 0, 0])])
    assert t["failures"] == {"missed": 1, "pressed": 0, "unlabelled": 1}
    assert t["jump_while_label_off"] == round(2 / 5, 3)


def test_a_pass_carried_by_one_level_does_not_count():
    rows = [_row([0], [0], level="Level5-1", best=900.0, rand=300.0) for _ in range(8)]
    for lv in ("Level8-1", "Level3-1", "Level6-1"):
        rows += [_row([0], [0], level=lv, best=290.0, rand=300.0) for _ in range(2)]
    t = fp.play_test(rows)
    assert t["pass"] and t["carried_by_one_level"] and not t["counts"]
    rows = [_row([0], [0], level=lv, best=500.0, rand=300.0) for lv in fp.LEVELS for _ in range(4)]
    t = fp.play_test(rows)
    assert t["pass"] and not t["carried_by_one_level"] and t["counts"]
