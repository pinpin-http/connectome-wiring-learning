"""Dense baselines. Parameter-matched to the graph policy, or the comparison
tells you nothing except which model is bigger."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPPolicy(nn.Module):
    """Feed-forward baseline with a short observation stack for context."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)
        self.n_actions = n_actions

    def reset_state(self, batch_size: int = 1, device=None) -> None:
        """Stateless; defined so every policy shares one interface."""

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: Tensor) -> int:
        return int(self(obs).argmax(dim=-1).item())


class GRUPolicy(nn.Module):
    """Recurrent dense baseline: isolates 'has memory' from 'has fly topology'."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 192):
        super().__init__()
        self.cell = nn.GRUCell(obs_dim, hidden)
        self.head = nn.Linear(hidden, n_actions)
        self.hidden = hidden
        self.h: Tensor | None = None
        self.n_actions = n_actions

    def reset_state(self, batch_size: int = 1, device=None) -> None:
        device = device or next(self.parameters()).device
        self.h = torch.zeros(batch_size, self.hidden, device=device)

    def detach_state(self) -> None:
        if self.h is not None:
            self.h = self.h.detach()

    def forward(self, obs: Tensor) -> Tensor:
        if self.h is None or self.h.shape[0] != obs.shape[0]:
            self.reset_state(obs.shape[0], obs.device)
        self.h = self.cell(obs, self.h)
        return self.head(self.h)

    @torch.no_grad()
    def act(self, obs: Tensor) -> int:
        return int(self(obs).argmax(dim=-1).item())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_hidden(obs_dim: int, n_actions: int, target_params: int) -> tuple[int, int]:
    """Pick MLP hidden widths whose parameter count is closest to `target_params`."""
    best, best_err = (64, 64), float("inf")
    for h in range(16, 1024, 8):
        n = (obs_dim + 1) * h + (h + 1) * h + (h + 1) * n_actions
        err = abs(n - target_params)
        if err < best_err:
            best, best_err = (h, h), err
    return best
