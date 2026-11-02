"""A schematic view of the game drawn from RAM: the "perfect perception" control.

A declared control condition, never the primary one (docs/PERCEPTION_CREDIT_STUDY.md). Same frame shape
and same eye (RetinotopicVision); white = empty, grey = non-empty metatiles, black = on-screen enemies
(16 x 16), dark = Mario. The frame is cropped 8 px on the left and at the top.
RAM: screen left page 0x71A, x 0x71C; level buffer 0x500-0x69F (two pages of 13 x 16, row r at y 32 + 16 r);
enemy i: active 0x0F+i, vertical page 0xB6+i, page 0x6E+i, x 0x87+i, y 0xCF+i; Mario: 0xB5, 0x6D, 0x86,
0xCE, size 0x754 (0 big, 1 small). 0x3AE+i is not a screen x for off-screen enemies.
"""
from __future__ import annotations

import numpy as np

WHITE, TILE, ENEMY, MARIO = 255, 128, 0, 60
CROP_X, CROP_Y = -8, -8


def screen_enemies(ram: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Active enemies on screen: (level x, screen x, screen y, type)."""
    ram = ram.astype(int)
    left = ram[0x71A] * 256 + ram[0x71C]
    out = []
    for i in range(5):
        lx = ram[0x6E + i] * 256 + ram[0x87 + i]
        if ram[0x0F + i] and ram[0xB6 + i] == 1 and -16 < lx - left < 256:
            out.append((lx, lx - left + CROP_X, int(ram[0xCF + i]), int(ram[0x16 + i])))
    return out


def danger_ahead(ram: np.ndarray) -> float:
    """Distance (px along the level) from Mario to the nearest on-screen enemy
    ahead of him; 999 if none. Stricter than `MarioEnv.enemy_ahead`, which also
    counts enemies off screen vertically (e.g. a Lakitu out of view)."""
    ram = ram.astype(int)
    mario = ram[0x6D] * 256 + ram[0x86]
    d = [lx - mario for lx, _, _, _ in screen_enemies(ram) if lx > mario]
    return float(min(d)) if d else 999.0


def schematic_frame(ram: np.ndarray, shape=(224, 240)) -> np.ndarray:
    """The schematic picture as an RGB uint8 frame of `shape` (H, W)."""
    ram = ram.astype(int)
    H, W = shape[:2]
    img = np.full((H, W), WHITE, np.uint8)

    def box(x0, y0, w, h, v):
        img[max(0, y0):min(H, y0 + h), max(0, x0):min(W, x0 + w)] = v

    left = ram[0x71A] * 256 + ram[0x71C]
    for k in range(17):
        lx = (left // 16 + k) * 16
        c = lx // 16
        for r in range(13):
            if ram[0x500 + (c // 16 % 2) * 0xD0 + r * 16 + (c % 16)]:
                box(lx - left + CROP_X, 32 + 16 * r + CROP_Y, 16, 16, TILE)
    for _, sx, sy, _ in screen_enemies(ram):
        box(sx, sy, 16, 16, ENEMY)
    if ram[0xB5] == 1:
        big = ram[0x754] == 0
        x0 = ram[0x6D] * 256 + ram[0x86] - left + CROP_X
        box(x0, ram[0xCE] + (0 if big else 16) + CROP_Y, 16, 32 if big else 16, MARIO)
    return np.repeat(img[..., None], 3, axis=2)


def _tile(ram: np.ndarray, lx: int, row: int) -> int:
    """Metatile of the level buffer at level x `lx`, buffer row `row` (0-12)."""
    if not 0 <= row < 13:
        return 0
    return int(ram[0x500 + ((lx // 256) % 2) * 0xD0 + row * 16 + (lx % 256) // 16])


def jump_reasons(ram: np.ndarray) -> dict:
    """Why a jump is called for now, from RAM (a teacher's label; never an input).

    enemy  an on-screen enemy 8-64 px ahead (`danger_ahead`)
    wall   a non-empty metatile at Mario's body height 4-28 px ahead of his front
    pit    no metatile under Mario's feet anywhere 8-40 px ahead (all rows below)
    Mario's box: x = page 0x6D, 0x86 (16 wide); feet at y 0xCE + 32 (small and big).
    """
    ram = ram.astype(int)
    if ram[0xB5] != 1:
        return {"enemy": False, "wall": False, "pit": False}
    mx = ram[0x6D] * 256 + ram[0x86]
    feet = ram[0xCE] + 32
    big = ram[0x754] == 0
    top = ram[0xCE] + (0 if big else 16)
    body_rows = sorted({(y - 32) // 16 for y in range(top, feet, 8)})
    feet_row = (feet - 32) // 16
    d = danger_ahead(ram)
    wall = any(_tile(ram, mx + 16 + dx, r) for dx in range(4, 29, 8) for r in body_rows)
    pit = any(not any(_tile(ram, mx + 8 + dx, r) for r in range(feet_row, 13)) for dx in range(8, 41, 8))
    return {"enemy": 8 <= d <= 64, "wall": bool(wall), "pit": bool(pit)}


def jump_label(ram: np.ndarray) -> bool:
    """The full teacher label: jump if an enemy, a wall or a pit is just ahead."""
    r = jump_reasons(ram)
    return r["enemy"] or r["wall"] or r["pit"]


def takeoff_label(ram: np.ndarray) -> bool:
    """Take-off label: jump only when the obstacle is at take-off
    distance -- enemy 16-40 px ahead, wall 4-12 px ahead of Mario's front, pit
    edge 8-16 px ahead -- so that a memoryless policy following it presses and
    releases by itself as the window moves past (a held A does not re-jump)."""
    ram = ram.astype(int)
    if ram[0xB5] != 1:
        return False
    mx = ram[0x6D] * 256 + ram[0x86]
    feet = ram[0xCE] + 32
    top = ram[0xCE] + (0 if ram[0x754] == 0 else 16)
    body_rows = sorted({(y - 32) // 16 for y in range(top, feet, 8)})
    feet_row = (feet - 32) // 16
    d = danger_ahead(ram)
    wall = any(_tile(ram, mx + 16 + dx, r) for dx in (4, 12) for r in body_rows)
    pit = any(not any(_tile(ram, mx + 8 + dx, r) for r in range(feet_row, 13)) for dx in (8, 16))
    return bool(16 <= d <= 40 or wall or pit)


#: enemy types never counted as a ground threat to jump over: Piranha Plant (0x0D,
#: hides in pipes), Lakitu (0x11, flies above); RAM types as in `screen_enemies`
NOT_GROUND_THREATS = {0x0D, 0x11}


def threat_ahead(ram: np.ndarray) -> float:
    """Distance (px) to the nearest on-screen enemy ahead that is a *ground threat*:
    vertically within Mario's body band (enemy top between his head - 16 px and his
    feet) and not a Piranha Plant or Lakitu; 999 if none."""
    ram = ram.astype(int)
    mario = ram[0x6D] * 256 + ram[0x86]
    feet = ram[0xCE] + 32
    head = ram[0xCE] + (0 if ram[0x754] == 0 else 16)
    d = [lx - mario for lx, _sx, sy, t in screen_enemies(ram)
         if lx > mario and t not in NOT_GROUND_THREATS and head - 16 <= sy + 8 <= feet]
    return float(min(d)) if d else 999.0


def takeoff_label_v2(ram: np.ndarray) -> bool:
    """`takeoff_label` with the enemy part restricted to ground threats (threat_ahead)."""
    ram = ram.astype(int)
    if ram[0xB5] != 1:
        return False
    mx = ram[0x6D] * 256 + ram[0x86]
    feet = ram[0xCE] + 32
    top = ram[0xCE] + (0 if ram[0x754] == 0 else 16)
    body_rows = sorted({(y - 32) // 16 for y in range(top, feet, 8)})
    feet_row = (feet - 32) // 16
    d = threat_ahead(ram)
    wall = any(_tile(ram, mx + 16 + dx, r) for dx in (4, 12) for r in body_rows)
    pit = any(not any(_tile(ram, mx + 8 + dx, r) for r in range(feet_row, 13)) for dx in (8, 16))
    return bool(16 <= d <= 40 or wall or pit)
