#!/usr/bin/env python
"""Study S follow-up F3 (docs/STUDY_S.md, declared 2026-10-12 06:32 UTC): the declared K = 20 reading of rung c next
to the K = 40 one, endpoint by endpoint (rank, z, adjusted p, verdict). Descriptive; the verdicts are those of
study_s_reading.py, unchanged. Writes artifacts/runs/study_s/f3_compare.txt.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "artifacts/runs/study_s"


def main():
    r20, r40 = (json.loads((D / f"reading-K{k}c.json").read_text()) for k in (20, 40))
    assert r20["rung_c"]["K"] == 20 and r40["rung_c"]["K"] == 40 and r40["rung_b"] == r20["rung_b"]
    rows = ["endpoint      | rung b (K 40): rank, z, adj p | rung c K 20: rank, z, adj p | rung c K 40: rank, z, adj p, null range | verdict K 20 -> K 40",
            "-" * 150]
    for fam in ("primary", "secondary"):
        for key, b in r40["rung_b"][fam].items():
            c20, c40 = r20["rung_c"][fam][key], r40["rung_c"][fam][key]
            rows.append(f"{key:13} | {b['rank_from_bottom']:2}/41 {b['z']:+6.2f} {b['p_adjusted_maxz']:.3f}      "
                        f"| {c20['rank_from_bottom']:2}/21 {c20['z']:+6.2f} {c20['p_adjusted_maxz']:.3f}      "
                        f"| {c40['rank_from_bottom']:2}/41 {c40['z']:+6.2f} {c40['p_adjusted_maxz']:.3f} "
                        f"[{c40['null_range'][0]:.3f}, {c40['null_range'][1]:.3f}] conn {c40['connectome']:.3f} "
                        f"| {r20['verdict'][key]} -> {r40['verdict'][key]}")
    # how the 20 added nulls compare with the first 20 (does the rung-c distribution move?)
    rows.append("")
    for t in ("where", "loom"):
        e = [json.loads((D / f"nullin{s}.json").read_text())["summary"][f"true/{t}"]["E1"] for s in range(40)]
        rows.append(f"E1/{t}: nullin 0-19 mean {sum(e[:20]) / 20:.4f} (max {max(e[:20]):.4f}); "
                    f"nullin 20-39 mean {sum(e[20:]) / 20:.4f} (max {max(e[20:]):.4f})")
    txt = "\n".join(rows)
    (D / "f3_compare.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
