"""The Gate G candidate learning rules (docs/GATE_G_PLAN.md §4), shared by the estimator check and the learner.

Each rule turns one tick of activity into an eligibility per plastic connection k (pre j -> motor neuron i,
magnitude m_k >= 0, fixed sign), in units of d(expected reward)/d(synapse count); the update is
`Δm = η (R − R̄) e`. Shapes follow TorchLIF: (connections, brains) and (motor neurons, brains).
(A) node perturbation: exact likelihood-ratio gradient for this model (GATE_G_PLAN §9, amendment 2).
(B) spike REINFORCE (escape noise) through the leak-filtered presynaptic trace; predicted biased.
(H) pre x post x reward: the biased Hebbian form, a negative reference only.
"""
from __future__ import annotations

import math

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


class Rule:
    """Eligibility accumulated over a window of ticks, per connection, per brain."""

    name = "rule"

    def __init__(self, post, sign, gain: float, sigma: float, decay: float,
                 threshold: float, batch: int, device):
        import torch
        self.torch = torch
        self.post = post                                  # (P,) long, local motor index
        self.P, self.B = int(post.numel()), int(batch)
        # d(input to i)/d(m_k) per presynaptic spike, as a column for broadcasting
        self.unit = (float(gain) * sign.float())[:, None]  # (P, 1)
        self.sigma, self.decay, self.theta = float(sigma), float(decay), float(threshold)
        self.e = torch.zeros((self.P, self.B), dtype=torch.float32, device=device)

    def reset_window(self) -> None:
        """A new decision starts: the eligibility restarts from zero."""
        self.e.zero_()

    def reset_state(self) -> None:
        """A new episode: forget everything, including any running trace."""
        self.e.zero_()

    def tick(self, pre_spk, xi, vhat, spk, free) -> None:  # pragma: no cover
        raise NotImplementedError

    def eligibility(self):
        """d(expected reward)/d(synapse count), per connection and brain."""
        return self.e * self.unit


class NodePerturbation(Rule):
    """(A) e_k = Σ_t gain sign_k s_j(t-1) ξ_i(t) / σ²."""

    name = "node_perturbation"

    def tick(self, pre_spk, xi, vhat, spk, free) -> None:
        self.e.addcmul_(pre_spk, xi.index_select(0, self.post), value=1.0 / self.sigma ** 2)


class NodePerturbationFullTrajectory(NodePerturbation):
    """Node perturbation with a cumulative score and one update per rollout.

    `e` is the log-likelihood score accumulated since the rollout started; weights stay fixed for
    the rollout and `finish_rollout()` is called after the summed update.
    """

    name = "node_perturbation_full_trajectory"
    full = True
    block_update = True
    block_len = 10

    def reset_window(self) -> None:
        """Keep the score across decision boundaries within a rollout."""

    def finish_rollout(self) -> None:
        """Forget the completed rollout's score after its single weight update."""
        self.e.zero_()


class SpikeReinforce(Rule):
    """(B) escape-noise score of the spike x leak-filtered presynaptic trace."""

    name = "spike_reinforce"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        torch = self.torch
        self.x = torch.zeros_like(self.e)                   # ∂v̂/∂m, in units of `unit`
        self.spk_prev = torch.zeros((int(self.post.max()) + 1, self.B), dtype=torch.bool,
                                    device=self.e.device)

    def reset_state(self) -> None:
        super().reset_state()
        self.x.zero_()
        self.spk_prev.zero_()

    def score(self, vhat, spk, free):
        """(s − p) φ(z) / (σ p (1 − p)), computed in log space: p(1−p) underflows
        when v̂ is far from threshold, and the ratio is an inverse Mills ratio."""
        torch = self.torch
        z = (self.theta - vhat) / self.sigma
        log_phi = -0.5 * z * z - _LOG_SQRT_2PI
        up = torch.exp(log_phi - torch.special.log_ndtr(-z))      # s = 1: φ/(1−Φ(z)) = φ/p
        down = -torch.exp(log_phi - torch.special.log_ndtr(z))    # s = 0: −φ/Φ(z) = −φ/(1−p)
        return torch.where(spk, up, down) * free / self.sigma

    def tick(self, pre_spk, xi, vhat, spk, free) -> None:
        # x(t) = decay (1 − s_i(t−1)) x(t−1) + s_j(t−1): the trace is reset by a
        # postsynaptic spike exactly as the membrane is.
        keep = (~self.spk_prev).float().index_select(0, self.post)
        self.x.mul_(keep).mul_(self.decay).add_(pre_spk)
        self.e.addcmul_(self.score(vhat, spk, free).index_select(0, self.post), self.x)
        self.spk_prev.copy_(spk)


class Hebbian(Rule):
    """(H) e_k = Σ_t sign_k s_j(t-1) s_i(t). Negative reference, never a candidate."""

    name = "pre_post_reward"

    def tick(self, pre_spk, xi, vhat, spk, free) -> None:
        self.e.addcmul_(pre_spk, spk.float().index_select(0, self.post))

    def eligibility(self):
        return self.e * (self.unit / self.unit.abs().clamp_min(1e-12))   # sign only


RULES = {r.name: r for r in (NodePerturbation, NodePerturbationFullTrajectory,
                             SpikeReinforce, Hebbian)}
