"""Static oriented gratings with two independent labels (mission §7).

Orientation convention (fixed here, illustrated in docs/retino_align/PILOT.md): the class gives the direction of
the **stripes** (the iso-luminance lines), measured counter-clockwise from the image's horizontal axis with y
pointing up. 0° = horizontal stripes (luminance varies along the vertical), 90° = vertical stripes. The angle is
defined in pixel space; the eye's column lattice is sampled anisotropically (6.4 px between nearest columns), so
the same angle is not the same angle on the hex lattice. That is part of the fixed interface, identical in every
condition and graph.

Image = base + centred(pattern), with

    pattern = A · env(r) · sin(2π ⟨n, p - c⟩ / period + phase) + noise

- env: raised cosine, 1 inside radius r0, 0 beyond r1, centred on the eye's sampling centroid plus a jitter:
  isotropic, so no orientation-dependent cropping, and inside the eye's footprint (covered up to 100 px).
- noise: i.i.d. uniform in [-noise, +noise] per pixel (bounded).
- centring: the pattern's mean over the eye's sample points, after the encoder's own blur, is subtracted before the
  brightness is added, so the mean luminance the eye sees is exactly `base` (mission §7.2).
- base = centre[brightness class] + U(-jitter, +jitter): the label is the nominal generator level; adjacent
  classes overlap when jitter > half the spacing (mission §7.2 variant).

Phase, position, period, contrast and noise are drawn independently of both labels. Every image has its own latent
draw; nothing is reused across images or partitions. Rendering is analytic at pixel centres, float (no 8-bit
quantisation), with periods of at least ~4 column spacings.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

H, W = 240, 256
HUD = 32
ORIENT_DEG = (0.0, 45.0, 90.0, 135.0)
BANKS = {"dev": 1, "pilot_train": 2, "pilot_val": 3, "pilot_test": 4, "train": 5, "val": 6, "test": 7}


@dataclass(frozen=True)
class GenParams:
    lum_centres: tuple = (0.35, 0.45, 0.55, 0.65)
    lum_jitter: float = 0.065
    amp_range: tuple = (0.05, 0.10)
    period_range: tuple = (24.0, 40.0)      # px
    noise: float = 0.02
    r0: float = 60.0                        # px, envelope plateau
    r1: float = 92.0                        # px, envelope edge
    pos_jitter: float = 8.0                 # px, envelope centre jitter (uniform, each axis)
    centre: tuple = (135.38, 124.16)        # (row, col) of the eye's sample centroid, frame coordinates

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bank:
    name: str
    uid: np.ndarray        # (n,) int64 unique example identifiers
    orient: np.ndarray     # (n,) 0..3
    lum: np.ndarray        # (n,) 0..3
    latent: dict = field(default_factory=dict)  # name -> (n,) array

    def __len__(self) -> int:
        return len(self.uid)


def draw_bank(name: str, per_combo: int, seed: int, gp: GenParams) -> Bank:
    """Latents of a balanced bank: `per_combo` images for each of the 16 (orientation, brightness) pairs, in a
    shuffled order. Labels are drawn first and the nuisance latents independently of them."""
    rng = np.random.default_rng(seed)
    combos = np.repeat(np.arange(16), per_combo)
    combos = combos[rng.permutation(len(combos))]
    n = len(combos)
    o, b = combos // 4, combos % 4
    lat = {
        "phase": rng.uniform(0.0, 2 * np.pi, n),
        "dy": rng.uniform(-gp.pos_jitter, gp.pos_jitter, n),
        "dx": rng.uniform(-gp.pos_jitter, gp.pos_jitter, n),
        "period": rng.uniform(*gp.period_range, n),
        "amp": rng.uniform(*gp.amp_range, n),
        "lum_offset": rng.uniform(-gp.lum_jitter, gp.lum_jitter, n),
        "noise_seed": rng.integers(0, 2**62, n, dtype=np.int64),
    }
    lat["base"] = np.asarray(gp.lum_centres)[b] + lat["lum_offset"]
    uid = BANKS[name] * 1_000_000 + np.arange(n, dtype=np.int64)
    return Bank(name=name, uid=uid, orient=o.astype(np.int8), lum=b.astype(np.int8), latent=lat)


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float64)


def sample_points(col_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (row, col) pixel, in the cropped image, that each column samples (vision.RetinotopicVision._sample)."""
    h, w = H - HUD, W
    rows = np.clip(((1.0 - col_uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    cols = np.clip((col_uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    return rows, cols


def render(bank: Bank, i: int, gp: GenParams, pts: tuple[np.ndarray, np.ndarray], blur_sigma: float) -> np.ndarray:
    """(H, W) float64 luminance in [0, 1] of image i."""
    L = bank.latent
    th = np.deg2rad(ORIENT_DEG[int(bank.orient[i])])
    nrm = (-np.sin(th), np.cos(th))                        # normal to the stripes, (x, y_up)
    cy, cx = gp.centre[0] + L["dy"][i], gp.centre[1] + L["dx"][i]
    x, y_up = _XX - cx, -(_YY - cy)
    r = np.hypot(x, y_up)
    env = 0.5 * (1.0 + np.cos(np.pi * np.clip((r - gp.r0) / (gp.r1 - gp.r0), 0.0, 1.0)))
    pat = L["amp"][i] * env * np.sin(2 * np.pi * (nrm[0] * x + nrm[1] * y_up) / L["period"][i] + L["phase"][i])
    pat += np.random.default_rng(int(L["noise_seed"][i])).uniform(-gp.noise, gp.noise, (H, W))
    blurred = cv2.GaussianBlur(pat[HUD:].astype(np.float32), (0, 0), blur_sigma) if blur_sigma > 0 else pat[HUD:]
    pat -= float(blurred[pts[0], pts[1]].astype(np.float64).mean())
    img = L["base"][i] + pat
    if img.min() < 0.0 or img.max() > 1.0:
        raise ValueError(f"image {bank.uid[i]} saturates: [{img.min():.3f}, {img.max():.3f}]")
    return img


def encode_bank(bank: Bank, gp: GenParams, vis, inj: np.ndarray, col_uv: np.ndarray,
                keep_images: int = 0) -> tuple[np.ndarray, dict]:
    """Static convention (mission §7.3): two identical frames after a reset, so the change term is exactly zero
    and only the static luminance term drives the eye. Returns X (n, N) float32 on `inj` and side products:
    a few images, the per-image sampled luminance at the columns, and image digests (for duplicate checks)."""
    pts = sample_points(col_uv)
    X = np.empty((len(bank), len(inj)), np.float32)
    col_lum = np.empty((len(bank), len(col_uv)), np.float32)
    digests = []
    imgs = []
    for i in range(len(bank)):
        img = render(bank, i, gp, pts, vis.p.blur_sigma)
        frame = np.repeat((img * 255.0).astype(np.float32)[:, :, None], 3, 2)
        vis.reset()
        vis.encode(frame)
        cur = vis.encode(frame)                            # second identical frame: delta = 0
        X[i] = cur[inj]
        gray = vis._luminance(frame)
        col_lum[i] = gray[pts[0], pts[1]]
        digests.append(hashlib.sha256(img.astype(np.float32).tobytes()).hexdigest()[:16])
        if i < keep_images:
            imgs.append(img.astype(np.float32))
    return X, {"col_lum": col_lum, "digests": np.array(digests), "images": np.array(imgs, np.float32)}
