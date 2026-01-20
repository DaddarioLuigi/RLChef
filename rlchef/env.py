from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, NamedTuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

Action = Literal["up", "down", "left", "right", "cook"]


@dataclass(frozen=True)
class Recipe:
    name: str
    need: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class EnvConfig:
    size: int = 5
    max_steps: int = 60
    pickup_reward: float = 0.05
    move_penalty: float = 0.01
    waste_penalty: float = 0.2
    fail_penalty: float = 0.5
    empty_cook_extra_penalty: float = 0.7
    incompat_penalty: float = 0.3
    step_penalty: float = 0.0
    # NOTE:
    # - "simple" is intentionally lossy (not Markov) and is kept to study the impact of state representation.
    # - "mask" includes the visited-cells mask and makes the tabular state Markov (given a fixed grid layout).
    state_mode: Literal["simple", "mask"] = "simple"
    # Grid layout:
    # - "fixed": deterministic 5x5 ingredient layout (recommended for reproducible results and Markov tabular state)
    # - "random": sample a new layout every episode (harder for tabular; can be used as a robustness experiment)
    grid_mode: Literal["fixed", "random"] = "fixed"
    # Challenging variant (budget / costs / multi-round wealth maximization)
    variant: Literal["base", "budget"] = "base"
    rounds: int = 1
    start_budget: float = 10.0
    # Cost per ingredient type (len == number of ingredients). If None, a default vector is used.
    ingredient_costs: tuple[float, ...] | None = None
    # If False, the agent cannot buy an ingredient when budget is insufficient.
    allow_debt: bool = False
    # If True, the map supply resets after each cook (new round). If False, supply is shared across rounds.
    reset_supply_each_round: bool = True


class StepOut(NamedTuple):
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class ChefGridEnv(gym.Env):
    """
    State: (x, y, inventory_counts...) -> float vector observation.
    Each cell sells one ingredient type (0..K-1) and can be collected only once.
    Actions: 4-dir movement + cook.
    Reward: small pickup reward, movement penalty, cook gives recipe value - waste - incompatibility penalties.
    """

    metadata = {"render_modes": ["human"], "render_fps": 12}

    def __init__(
        self,
        *,
        config: EnvConfig,
        ingredient_names: list[str],
        recipes: list[Recipe],
        incompat_pairs: Iterable[tuple[int, int]] = (),
        seed: int | None = None,
        render_mode: str | None = None,
        fixed_grid: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.ingredient_names = ingredient_names
        self.recipes = recipes
        self.k = len(ingredient_names)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self._fixed_grid = fixed_grid.copy() if fixed_grid is not None else None

        # Budget-variant parameters
        if self.cfg.variant == "budget":
            if self.cfg.rounds < 1:
                raise ValueError("rounds must be >= 1")
            if self.cfg.start_budget < 0:
                raise ValueError("start_budget must be >= 0")
            if self.cfg.ingredient_costs is not None and len(self.cfg.ingredient_costs) != self.k:
                raise ValueError("ingredient_costs length must match number of ingredients")
            self._costs = tuple(self.cfg.ingredient_costs) if self.cfg.ingredient_costs is not None else None
        else:
            self._costs = None

        self.action_space = spaces.Discrete(self.n_actions)
        base_dim = 3 + 3 * self.k  # pos(2) + cell_onehot(k) + available(1) + inv_norm(k) + remaining_norm(k)
        obs_dim = base_dim + (2 if self.cfg.variant == "budget" else 0)  # + budget_norm + round_norm
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        incompat = np.zeros((self.k, self.k), dtype=np.int8)
        for a, b in incompat_pairs:
            incompat[a, b] = 1
            incompat[b, a] = 1
        self.incompat = incompat

        self._grid: np.ndarray | None = None
        self._available: np.ndarray | None = None
        self._pos: tuple[int, int] | None = None
        self._inv: np.ndarray | None = None
        self._steps: int = 0
        self._round: int = 0
        self._budget: float = 0.0
        self._viewer = None
        self._last_reward: float = 0.0
        self._last_made: str | None = None

    @property
    def n_actions(self) -> int:
        return 5

    def action_index(self, a: Action) -> int:
        return {"up": 0, "down": 1, "left": 2, "right": 3, "cook": 4}[a]

    def index_action(self, i: int) -> Action:
        return ["up", "down", "left", "right", "cook"][i]  # type: ignore[return-value]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if hasattr(self, "np_random") and self.np_random is not None:
            # Gymnasium seeding sets self.np_random (np.random.Generator)
            self.rng = self.np_random  # type: ignore[assignment]
        s = self.cfg.size
        if self.cfg.grid_mode == "fixed":
            if self._fixed_grid is None:
                raise ValueError("grid_mode='fixed' requires fixed_grid")
            if self._fixed_grid.shape != (s, s):
                raise ValueError(f"fixed_grid must have shape {(s, s)}, got {self._fixed_grid.shape}")
            self._grid = self._fixed_grid.copy()
        else:
            self._grid = self.rng.integers(low=0, high=self.k, size=(s, s), dtype=np.int64)
        self._available = np.ones((s, s), dtype=np.int8)
        self._pos = (int(self.rng.integers(0, s)), int(self.rng.integers(0, s)))
        self._inv = np.zeros(self.k, dtype=np.int64)
        self._steps = 0
        self._round = 0
        self._budget = float(self.cfg.start_budget) if self.cfg.variant == "budget" else 0.0
        self._last_reward = 0.0
        self._last_made = None
        return self._obs(), {
            "pos": self._pos,
            "grid": self._grid.copy(),
            "grid_mode": self.cfg.grid_mode,
            "variant": self.cfg.variant,
            "round": int(self._round),
            "budget": float(self._budget),
        }

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._grid is not None and self._available is not None
        assert self._pos is not None and self._inv is not None

        a = self.index_action(action_idx)
        r = -self.cfg.step_penalty
        terminated = False
        truncated = False
        info: dict = {"variant": self.cfg.variant, "round": int(self._round), "budget": float(self._budget)}

        if a == "cook":
            cook_reward, cook_info = self._cook()
            r += cook_reward
            info.update(cook_info)

            if self.cfg.variant == "budget":
                # End of round; possibly continue to next round
                self._round += 1
                info["round"] = int(self._round)
                info["budget"] = float(self._budget)
                if self._round >= self.cfg.rounds:
                    terminated = True
                else:
                    # New round: reset inventory and (optionally) supply and position
                    assert self._inv is not None
                    self._inv[:] = 0
                    if self.cfg.reset_supply_each_round:
                        assert self._available is not None
                        self._available[:] = 1
                    s = self.cfg.size
                    self._pos = (int(self.rng.integers(0, s)), int(self.rng.integers(0, s)))
            else:
                terminated = True
        else:
            r -= self.cfg.move_penalty
            x, y = self._pos
            nx, ny = x, y
            if a == "up":
                nx = max(0, x - 1)
            elif a == "down":
                nx = min(self.cfg.size - 1, x + 1)
            elif a == "left":
                ny = max(0, y - 1)
            elif a == "right":
                ny = min(self.cfg.size - 1, y + 1)
            self._pos = (nx, ny)

            if self._available[nx, ny] == 1:
                ing = int(self._grid[nx, ny])
                if self.cfg.variant == "budget":
                    cost = float(self._ingredient_cost(ing))
                    can_buy = self.cfg.allow_debt or (self._budget >= cost)
                    if can_buy:
                        self._inv[ing] += 1
                        self._available[nx, ny] = 0
                        self._budget -= cost
                        r += self.cfg.pickup_reward - cost
                        info.update({"picked": ing, "cost": cost, "budget": float(self._budget)})
                    else:
                        # Cannot afford: do not buy, keep availability
                        info.update({"picked": None, "cost": cost, "could_not_afford": True, "budget": float(self._budget)})
                else:
                    self._inv[ing] += 1
                    self._available[nx, ny] = 0
                    r += self.cfg.pickup_reward

        self._steps += 1
        if self._steps >= self.cfg.max_steps and not terminated:
            truncated = True
            info["timeout"] = True

        obs = self._obs()
        self._last_reward = float(r)
        self._last_made = info.get("made") if terminated else None
        return obs, float(r), terminated, truncated, info

    def render(self, *, last_reward: float | None = None, made: str | None = None, fps: int | None = None) -> bool:
        assert self._grid is not None and self._available is not None
        assert self._pos is not None and self._inv is not None
        if self._viewer is None:
            from rlchef.pygame_viewer import PygameViewer

            self._viewer = PygameViewer(size=self.cfg.size, k=self.k, ingredient_names=self.ingredient_names)
        if last_reward is None:
            last_reward = self._last_reward
        if made is None:
            made = self._last_made
        ok = bool(
            self._viewer.render(
                grid=self._grid,
                available=self._available,
                pos=self._pos,
                inv=self._inv,
                last_reward=last_reward,
                made=made,
                step=self._steps,
                max_steps=self.cfg.max_steps,
                budget=(float(self._budget) if self.cfg.variant == "budget" else None),
                round_idx=(int(self._round) if self.cfg.variant == "budget" else None),
                rounds_total=(int(self.cfg.rounds) if self.cfg.variant == "budget" else None),
            )
        )
        self._viewer.tick(fps)
        return ok

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _cook(self) -> tuple[float, dict]:
        inv = self._inv
        assert inv is not None

        empty = bool(inv.sum() == 0)

        best: Recipe | None = None
        best_value = -1e18
        for rec in self.recipes:
            if self._can_make(inv, rec.need):
                v = rec.value
                if v > best_value:
                    best_value = v
                    best = rec

        incompat_pen = self._incompat_penalty(inv)

        if best is None:
            extra = self.cfg.empty_cook_extra_penalty if empty else 0.0
            reward = -self.cfg.fail_penalty - extra - incompat_pen
            if self.cfg.variant == "budget":
                # Fail -> no revenue, but still "spent" by collecting ingredients already happened
                pass
            return (
                float(reward),
                {
                    "made": None,
                    "base_value": 0.0,
                    "waste": int(inv.sum()),
                    "incompat": float(incompat_pen),
                    "empty_cook": empty,
                    "revenue": 0.0,
                    "budget": float(self._budget),
                },
            )

        used = np.array(best.need, dtype=np.int64)
        waste = int((inv - used).clip(min=0).sum())
        waste_pen = self.cfg.waste_penalty * float(waste)
        total = best.value - waste_pen - incompat_pen
        revenue = float(max(0.0, total)) if self.cfg.variant == "budget" else float(total)
        if self.cfg.variant == "budget":
            self._budget += revenue

        return (
            float(revenue),
            {
                "made": best.name,
                "base_value": float(best.value),
                "waste": waste,
                "incompat": float(incompat_pen),
                "empty_cook": empty,
                "revenue": float(revenue),
                "budget": float(self._budget),
            },
        )

    def _can_make(self, inv: np.ndarray, need: tuple[int, ...]) -> bool:
        if len(need) != self.k:
            raise ValueError("Recipe.need deve avere lunghezza == numero ingredienti")
        return bool(np.all(inv >= np.array(need, dtype=np.int64)))

    def _incompat_penalty(self, inv: np.ndarray) -> float:
        present = np.where(inv > 0)[0]
        if present.size <= 1:
            return 0.0
        pen = 0.0
        for i in range(present.size):
            for j in range(i + 1, present.size):
                if self.incompat[present[i], present[j]] == 1:
                    pen += self.cfg.incompat_penalty
        return pen

    def _obs(self) -> np.ndarray:
        assert self._pos is not None and self._inv is not None
        assert self._grid is not None and self._available is not None
        x, y = self._pos
        pos = np.array([x / (self.cfg.size - 1), y / (self.cfg.size - 1)], dtype=np.float32)
        inv = self._inv.astype(np.float32)
        inv_norm = inv / max(1.0, float(self.cfg.size * self.cfg.size))
        ing = int(self._grid[x, y])
        cell_onehot = np.zeros(self.k, dtype=np.float32)
        cell_onehot[ing] = 1.0
        available = np.array([float(self._available[x, y])], dtype=np.float32)

        remaining = np.zeros(self.k, dtype=np.float32)
        for i in range(self.k):
            remaining[i] = float(np.sum((self._grid == i) & (self._available == 1)))
        remaining_norm = remaining / max(1.0, float(self.cfg.size * self.cfg.size))

        base = np.concatenate([pos, cell_onehot, available, inv_norm, remaining_norm], dtype=np.float32)
        if self.cfg.variant != "budget":
            return base
        # Budget / round features
        # Normalize by a conservative upper bound to keep in [0,1] for typical settings.
        budget_norm = np.array([float(np.clip(self._budget / max(1e-6, self.cfg.start_budget), 0.0, 2.0)) / 2.0], dtype=np.float32)
        round_norm = np.array([float(self._round) / max(1.0, float(self.cfg.rounds))], dtype=np.float32)
        return np.concatenate([base, budget_norm, round_norm], dtype=np.float32)

    def state_key(self) -> tuple[int, int, tuple[int, ...]]:
        assert self._pos is not None and self._inv is not None
        x, y = self._pos
        if self.cfg.state_mode == "simple":
            key = (x, y, tuple(int(v) for v in self._inv.tolist()))
            if self.cfg.variant == "budget":
                return (*key, self._budget_bucket(), int(self._round))  # type: ignore[return-value]
            return key

        assert self._available is not None
        mask = 0
        s = self.cfg.size
        for i in range(s):
            for j in range(s):
                if self._available[i, j] == 0:
                    mask |= 1 << (i * s + j)
        key = (x, y, tuple(int(v) for v in self._inv.tolist()), mask)
        if self.cfg.variant == "budget":
            return (*key, self._budget_bucket(), int(self._round))  # type: ignore[return-value]
        return key

    def _ingredient_cost(self, ing: int) -> float:
        if self._costs is None:
            # Default costs (heuristic): aligned with default ingredient order
            return float([1.0, 1.2, 1.5, 0.8, 2.0][ing % 5])
        return float(self._costs[ing])

    def _budget_bucket(self) -> int:
        # Discretize budget for tabular keys (0.1 currency unit buckets).
        return int(np.clip(np.round(self._budget * 10.0), -10_000, 10_000))


def default_env(*, seed: int | None = None, config: EnvConfig | None = None) -> ChefGridEnv:
    names = ["tomato", "cheese", "pasta", "basil", "fish"]
    recipes = [
        Recipe("margherita", (1, 1, 0, 1, 0), 4.0),
        Recipe("pasta_al_pomodoro", (1, 0, 1, 1, 0), 3.5),
        Recipe("cheesy_pasta", (0, 1, 1, 0, 0), 3.0),
        Recipe("fish_special", (0, 0, 0, 1, 1), 3.2),
    ]
    incompat = [(2, 4)]  # pasta + fish
    cfg = config or EnvConfig()
    if cfg.variant == "budget" and cfg.ingredient_costs is None:
        # Default costs aligned with ingredient_names
        object.__setattr__(cfg, "ingredient_costs", (1.0, 1.2, 1.5, 0.8, 2.0))
    fixed_grid = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1],
            [3, 4, 0, 1, 2],
            [4, 0, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    return ChefGridEnv(
        config=cfg,
        ingredient_names=names,
        recipes=recipes,
        incompat_pairs=incompat,
        seed=seed,
        fixed_grid=fixed_grid,
    )

