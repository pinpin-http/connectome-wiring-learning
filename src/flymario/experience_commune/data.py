"""Shared, immutable inputs of one block (docs/experience_commune/PROTOCOL_FROZEN.md §3): every graph and mode reads them.

Per block and task (`where`, the T-field task, and `loom`, approach or recede: the Study S A1 stimuli of
`scripts/diagnostics/study_s.py`):

    train_x, train_cls, train_uid      6,000 training stimuli, classes balanced in consecutive blocks of 8
    U_action, U_reward                 the common-experience journal: actions drawn uniformly with replacement,
                                       independently of the graph, the scores and the labels; reward = action correct
    N_uniform                          one uniform per decision, shared by every graph, for sampling N's own policy
    progress_*                         512 evaluation images (64 per class), the same at every checkpoint
    final_*                            2,048 independent final images (256 per class)
    ood_*                              512 images of Study S's generalisation variants (64 per class)
    seeds                              every seed used, derived from SeedSequence([20261020, block, task, purpose])

The block also fixes the readout projection seed, the training noise seed (one stream shared by N and U and by every
graph) and the evaluation noise seed (keyed per image, independent of training).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = 20261020
TASKS = ("where", "loom")
BLOCK_CODE = {"b0": 1, "b1": 2, "b2": 3, "b3": 4, "b4": 5, "b5": 6, "d0": 9,
              # REVISION48 (2026-10-24): six fresh main blocks, two tuning blocks, one development block; new codes,
              # so new SeedSequence draws, independent of EC's blocks
              "r0": 41, "r1": 42, "r2": 43, "r3": 44, "r4": 45, "r5": 46, "t0": 51, "t1": 52, "rd0": 59}
PURPOSE = {"train_cls": 1, "train_jit": 2, "U_action": 3, "N_uniform": 4, "noise_train": 5, "progress": 6,
           "final": 7, "ood": 8, "eval_noise": 9, "readout": 10, "shuffled_reward": 11}
BANK_CODE = {"train": 1, "progress": 2, "final": 3, "ood": 4}
N_TRAIN, N_PROGRESS_PER_CLASS, N_FINAL_PER_CLASS, N_OOD_PER_CLASS = 6000, 64, 256, 64


def seed_of(block: str, task: str | None, purpose: str) -> int:
    t = -1 if task is None else TASKS.index(task)
    ss = np.random.SeedSequence([STUDY, BLOCK_CODE[block], t + 1, PURPOSE[purpose]])
    return int(ss.generate_state(1, np.uint64)[0] % (2**62))


def uid(block: str, task: str, bank: str, i: np.ndarray) -> np.ndarray:
    return (BLOCK_CODE[block] * 10**7 + TASKS.index(task) * 10**6 + BANK_CODE[bank] * 10**5
            + np.asarray(i, np.int64))


def balanced_stream(n: int, seed: int) -> np.ndarray:
    """Classes in consecutive shuffled blocks of 8 (Study S's rule, study_s.class_stream)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(8) for _ in range(-(-n // 8))])[:n].astype(np.int8)


def bank_classes(per_class: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(8), per_class)).astype(np.int8)


def ood_kwargs(task: str) -> list[dict]:
    """Study S's generalisation variants (study_s.build_stimuli `gens`)."""
    return ([dict(size=(48, 80)), dict(lum=0.5), dict(jit=16)] if task == "where" else
            [dict(loom_radii=(24, 72)), dict(lum=0.5), dict(loom_radii=(40, 110))])


def build_block_task(block: str, task: str, vis, idx: np.ndarray, stimulus) -> dict:
    """Every array of one block and task. `stimulus(task, cls, rng, **kw)` is study_s.stimulus."""
    def enc(prev, cur):
        vis.reset()
        vis.encode(prev)
        return vis.encode(cur)[idx].astype(np.float32)

    seeds = {p: seed_of(block, task, p) for p in PURPOSE if p not in ("readout",)}
    seeds["readout"] = seed_of(block, None, "readout")
    out: dict = {}
    cls = balanced_stream(N_TRAIN, seeds["train_cls"])
    rng = np.random.default_rng(seeds["train_jit"])
    out["train_cls"], out["train_uid"] = cls, uid(block, task, "train", np.arange(N_TRAIN))
    out["train_x"] = np.stack([enc(*stimulus(task, int(c), rng)) for c in cls])
    a = np.random.default_rng(seeds["U_action"]).integers(0, 8, N_TRAIN).astype(np.int8)
    out["U_action"], out["U_reward"] = a, (a == cls).astype(np.int8)
    out["N_uniform"] = np.random.default_rng(seeds["N_uniform"]).random(N_TRAIN)
    out["shuffled_reward_uniform"] = np.random.default_rng(seeds["shuffled_reward"]).random(N_TRAIN)
    for bank, per in (("progress", N_PROGRESS_PER_CLASS), ("final", N_FINAL_PER_CLASS)):
        bc = bank_classes(per, seeds[bank])
        rng = np.random.default_rng(seeds[bank] + 1)
        out[f"{bank}_cls"], out[f"{bank}_uid"] = bc, uid(block, task, bank, np.arange(len(bc)))
        out[f"{bank}_x"] = np.stack([enc(*stimulus(task, int(c), rng)) for c in bc])
    bc = bank_classes(N_OOD_PER_CLASS, seeds["ood"])
    var = np.zeros(len(bc), np.int8)
    for c in range(8):
        m = np.flatnonzero(bc == c)
        var[m] = np.arange(len(m)) % 3
    rng = np.random.default_rng(seeds["ood"] + 1)
    kws = ood_kwargs(task)
    out["ood_cls"], out["ood_variant"] = bc, var
    out["ood_uid"] = uid(block, task, "ood", np.arange(len(bc)))
    out["ood_x"] = np.stack([enc(*stimulus(task, int(c), rng, **kws[int(v)])) for c, v in zip(bc, var)])
    out["seeds_json"] = np.array(json.dumps(seeds))
    out["eval_noise_seed"] = np.int64(seeds["eval_noise"])
    out["noise_train_seed"] = np.int64(seeds["noise_train"] % (2**31 - 1))
    out["readout_seed"] = np.int64(seeds["readout"] % (2**31 - 1))
    return out


def journal_csv(d: dict) -> str:
    """The U journal in the protocol's format."""
    lines = ["trial_id,stimulus_id,action_forced,behavior_probability,reward"]
    for k in range(len(d["U_action"])):
        lines.append(f"{k},{int(d['train_uid'][k])},{int(d['U_action'][k])},0.125,{int(d['U_reward'][k])}")
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
