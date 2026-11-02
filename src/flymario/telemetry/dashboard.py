"""Live three-panel view: the game, the brain, and the telemetry.

    +----------------+---------------------------+----------------+
    |  SUPER MARIO   |   CONNECTOME (spiking)    |   TELEMETRY    |
    |  NES output    |   real 3D anatomy,        |   actions      |
    |                |   projected to 2D         |   firing rate  |
    +----------------+---------------------------+----------------+

3D coordinates are projected to pixel indices once at startup; each frame only the neurons
that spiked are scattered into the image buffer, so cost scales with spikes, not brain size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

# Class -> RGB. Deliberately few colours: the eye should read
# "input / processing / output", not a taxonomy.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "optic": (70, 130, 255),
    "sensory": (70, 200, 255),
    "visual_projection": (120, 160, 255),
    "central": (235, 235, 235),
    "ascending": (255, 200, 90),
    "descending": (255, 80, 80),
    "motor": (255, 60, 120),
}
DEFAULT_COLOUR = (150, 150, 150)

BG = (10, 11, 14)
PANEL = (18, 20, 26)
TEXT = (225, 228, 235)
DIM = (120, 128, 145)
ACCENT = (255, 170, 60)


@dataclass
class Layout:
    game_w: int = 480          # 240 * 2
    game_h: int = 448          # 224 * 2
    brain_w: int = 560
    tele_w: int = 300
    pad: int = 12
    header: int = 34

    @property
    def width(self) -> int:
        return self.pad * 4 + self.game_w + self.brain_w + self.tele_w

    @property
    def height(self) -> int:
        return self.pad * 2 + self.header + max(self.game_h, 520)


class Dashboard:
    def __init__(self, graph, motor_idx: np.ndarray, action_names: list[str],
                 layout: Layout | None = None, title: str = "FlyMario"):
        self.L = layout or Layout()
        self.action_names = action_names
        self.motor_idx = motor_idx

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.L.width, self.L.height))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("monospace", 17, bold=True)
        self.f = pygame.font.SysFont("monospace", 13)
        self.f_small = pygame.font.SysFont("monospace", 11)

        self._project(graph)
        self.brain_buf = np.zeros((self.bh, self.bw, 3), dtype=np.float32)
        self.rate_hist: list[float] = []
        self.running = True

    # -- one-off projection ------------------------------------------------
    def _project(self, graph) -> None:
        """3D EM coordinates -> pixel indices in the brain panel, computed once."""
        self.bw = self.L.brain_w
        self.bh = self.L.height - self.L.pad * 2 - self.L.header
        pos = np.asarray(graph.node_pos, dtype=np.float32)
        # x is the left-right axis of the brain, y is dorsoventral: that pair
        # gives the familiar frontal view of the fly brain.
        xy = pos[:, [0, 1]]
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        norm = (xy - lo) / span
        m = 10
        self.px = (norm[:, 0] * (self.bw - 2 * m) + m).astype(np.int32)
        self.py = ((1.0 - norm[:, 1]) * (self.bh - 2 * m) + m).astype(np.int32)

        classes = np.asarray(graph.node_class)
        self.colour = np.array(
            [CLASS_COLOURS.get(str(c), DEFAULT_COLOUR) for c in classes], dtype=np.float32
        )
        self.counts = {k: int((classes == k).sum()) for k in CLASS_COLOURS
                       if (classes == k).sum() > 0}
        self.is_motor = np.isin(classes, ("descending", "motor",
                                          "descending_neuron", "vnc_motor", "cb_motor"))

    # -- per-frame ---------------------------------------------------------
    def pump(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.running = False
        return self.running

    def _panel(self, x: int, y: int, w: int, h: int, title: str) -> None:
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=6)
        self.screen.blit(self.f_small.render(title, True, DIM), (x + 8, y + 6))

    def _draw_brain(self, spikes: np.ndarray, x0: int, y0: int) -> None:
        self.brain_buf *= 0.82                      # phosphor-style decay
        if spikes is not None and spikes.any():
            idx = np.flatnonzero(spikes)
            np.add.at(self.brain_buf, (self.py[idx], self.px[idx]), self.colour[idx])
            # Descending and motor neurons are ~1% of the brain and vanish among
            # 77k optic cells, yet they are the output we care about. Draw the
            # active ones again, brighter and as a 3x3 blob.
            hot = idx[self.is_motor[idx]]
            if hot.size:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = np.clip(self.py[hot] + dy, 0, self.bh - 1)
                        xx = np.clip(self.px[hot] + dx, 0, self.bw - 1)
                        np.add.at(self.brain_buf, (yy, xx), self.colour[hot] * 1.6)
        img = np.clip(self.brain_buf, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self.screen.blit(surf, (x0, y0))

    def _bar(self, x: int, y: int, w: int, h: int, frac: float,
             colour=ACCENT, label: str = "") -> None:
        pygame.draw.rect(self.screen, (40, 44, 55), (x, y, w, h), border_radius=3)
        fw = int(max(0.0, min(1.0, frac)) * w)
        if fw > 0:
            pygame.draw.rect(self.screen, colour, (x, y, fw, h), border_radius=3)
        if label:
            self.screen.blit(self.f_small.render(label, True, TEXT), (x + 4, y + h // 2 - 7))

    def render(self, frame: np.ndarray, spikes: np.ndarray, motor_trace: np.ndarray,
               action: int, stats: dict) -> None:
        L = self.L
        self.screen.fill(BG)
        self.screen.blit(self.f_big.render(
            f"FlyMario  ·  {stats.get('source', '?')}  ·  {stats.get('topology', '?')}",
            True, TEXT), (L.pad, 8))

        y = L.pad + L.header
        # --- game ---
        x = L.pad
        self._panel(x, y, L.game_w, L.game_h + 22, "NES")
        surf = pygame.surfarray.make_surface(np.transpose(frame[:, :, :3], (1, 0, 2)))
        self.screen.blit(pygame.transform.scale(surf, (L.game_w, L.game_h)), (x, y + 20))

        # --- brain ---
        x += L.game_w + L.pad
        bh = L.height - L.pad * 2 - L.header
        self._panel(x, y, L.brain_w, bh, f"CONNECTOME — {stats.get('neurons', 0):,} neurons")
        self._draw_brain(spikes, x, y)
        lx, ly = x + 10, y + bh - 16 * (len(self.counts) + 1)
        for k, n in self.counts.items():
            pygame.draw.rect(self.screen, CLASS_COLOURS[k], (lx, ly + 3, 9, 9))
            self.screen.blit(self.f_small.render(f"{k}  {n:,}", True, DIM), (lx + 15, ly))
            ly += 16

        # --- telemetry ---
        x += L.brain_w + L.pad
        self._panel(x, y, L.tele_w, bh, "TELEMETRY")
        ty = y + 26
        self.screen.blit(self.f.render("ACTION", True, DIM), (x + 10, ty)); ty += 18
        for i, name in enumerate(self.action_names):
            sel = i == action
            pygame.draw.rect(self.screen, ACCENT if sel else (40, 44, 55),
                             (x + 10, ty, 12, 12), border_radius=2)
            self.screen.blit(self.f.render(name, True, TEXT if sel else DIM), (x + 30, ty - 2))
            ty += 17

        ty += 10
        rate = stats.get("pop_rate", 0.0)
        self.screen.blit(self.f.render("FIRING RATE", True, DIM), (x + 10, ty)); ty += 18
        self._bar(x + 10, ty, L.tele_w - 20, 16, rate / 0.15, (90, 200, 140),
                  f"{rate * 100:.2f}% pop"); ty += 22
        mrate = stats.get("motor_rate", 0.0)
        self._bar(x + 10, ty, L.tele_w - 20, 16, mrate / 0.15, (255, 80, 80),
                  f"{mrate * 100:.2f}% motor"); ty += 30

        # firing-rate history
        self.rate_hist.append(rate)
        self.rate_hist = self.rate_hist[-(L.tele_w - 20):]
        gx, gy, gh = x + 10, ty, 46
        pygame.draw.rect(self.screen, (28, 31, 39), (gx, gy, L.tele_w - 20, gh), border_radius=3)
        if len(self.rate_hist) > 1:
            top = max(0.02, max(self.rate_hist))
            pts = [(gx + i, gy + gh - int(v / top * (gh - 4)) - 2)
                   for i, v in enumerate(self.rate_hist)]
            pygame.draw.lines(self.screen, (90, 200, 140), False, pts, 1)
        ty += gh + 16

        for key, fmt in (("mario_x", "x        {:>8.0f}"), ("reward", "reward   {:>8.1f}"),
                         ("episode_reward", "cum rew  {:>8.1f}"), ("frame", "frame    {:>8d}"),
                         ("fps", "fps      {:>8.0f}")):
            if key in stats:
                self.screen.blit(self.f.render(fmt.format(stats[key]), True, TEXT), (x + 10, ty))
                ty += 17

        motor_on = int((motor_trace > 0.01).sum()) if motor_trace is not None else 0
        ty += 8
        self.screen.blit(self.f_small.render(
            f"motor features active {motor_on}/{len(self.motor_idx)}", True, DIM), (x + 10, ty))

        pygame.display.flip()

    def tick(self, fps: int = 60) -> float:
        self.clock.tick(fps)
        return self.clock.get_fps()

    def close(self) -> None:
        pygame.quit()
