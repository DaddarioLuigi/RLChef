from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int: ...

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        *,
        next_action: int | None = None,
    ) -> None: ...


def _eps_greedy(rng: np.random.Generator, q: np.ndarray, eps: float) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, q.size))
    return int(np.argmax(q))


@dataclass
class TabularAgent:
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    eps: float = 0.1
    kind: str = "qlearning"  # or "sarsa"
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.q: dict[tuple, np.ndarray] = {}
        self._eps_min = float(self.eps)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)
        self._eps_min = min(self._eps_min, self.eps)

    def _q(self, key: tuple) -> np.ndarray:
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[key]

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        raise RuntimeError("TabularAgent requires a discrete key: use act_key/update_key")

    def act_key(self, key: tuple, *, greedy: bool = False) -> int:
        q = self._q(key)
        if greedy:
            return int(np.argmax(q))
        return _eps_greedy(self.rng, q, self.eps)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        *,
        next_action: int | None = None,
    ) -> None:
        raise RuntimeError("TabularAgent requires a discrete key: use update_key")

    def update_key(
        self,
        key: tuple,
        action: int,
        reward: float,
        next_key: tuple,
        terminated: bool,
        truncated: bool,
        *,
        next_action: int | None = None,
    ) -> None:
        q = self._q(key)
        target = reward
        if not (terminated or truncated):
            nq = self._q(next_key)
            if self.kind == "sarsa":
                if next_action is None:
                    raise ValueError("SARSA requires next_action")
                target += self.gamma * float(nq[next_action])
            else:
                target += self.gamma * float(np.max(nq))
        q[action] += self.alpha * (target - float(q[action]))


@dataclass
class LinearQAgent:
    n_actions: int
    obs_dim: int
    alpha: float = 0.05
    gamma: float = 0.99
    eps: float = 0.1
    l2: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.w = self.rng.normal(0.0, 0.01, size=(self.n_actions, self.obs_dim)).astype(np.float32)
        self.b = np.zeros(self.n_actions, dtype=np.float32)
        self._eps_min = float(self.eps)

    def set_eps(self, eps: float) -> None:
        self.eps = float(eps)
        self._eps_min = min(self._eps_min, self.eps)

    def q(self, obs: np.ndarray) -> np.ndarray:
        return (self.w @ obs.astype(np.float32)) + self.b

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        q = self.q(obs)
        if greedy:
            return int(np.argmax(q))
        return _eps_greedy(self.rng, q, self.eps)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        *,
        next_action: int | None = None,
    ) -> None:
        obs = obs.astype(np.float32)
        next_obs = next_obs.astype(np.float32)
        q = self.q(obs)
        target = reward
        if not (terminated or truncated):
            target += self.gamma * float(np.max(self.q(next_obs)))
        td = target - float(q[action])

        grad_w = obs
        self.w[action] += self.alpha * (td * grad_w - self.l2 * self.w[action])
        self.b[action] += self.alpha * td

