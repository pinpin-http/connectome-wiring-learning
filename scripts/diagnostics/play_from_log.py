#!/usr/bin/env python
"""Partial rate-matched play results from a closed_loop_rate_check log, for a run cut
by a deadline (2026-10-06: the evaluations were bounded at 21:37). Each log line is
one start (level, delay): <learner[:5]><last char>=<best x>/<mean x of the 5 random
jumpers>. Same test as the script (one-sided Wilcoxon, learner > random) on the
starts completed; labelled PARTIAL.

    .venv/bin/python scripts/diagnostics/play_from_log.py artifacts/logs/nullcmp_play0.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gates"))


def main():
    from oracle_criterion_check import criterion
    rows = {}
    n = 0
    for line in Path(sys.argv[1]).read_text().splitlines():
        if not re.match(r"^Level\d-\d delay", line):
            continue
        n += 1
        for name, x, r in re.findall(r"(\S+)=([\d.]+)/([\d.na]+)", line.split(":", 1)[1]):
            if r != "nan":
                rows.setdefault(name, []).append((float(x), float(r)))
    print(f"PARTIAL: {n} of 32 starts")
    for name, v in rows.items():
        a = np.array(v)
        print(name, criterion(a[:, 0], a[:, 1]))


if __name__ == "__main__":
    main()
