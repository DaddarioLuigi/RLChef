from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from rlchef.agents import LinearQAgent, TabularAgent
from rlchef.env import EnvConfig, default_env


@dataclass
class RunSummary:
    algo: str
    state_mode: str
    seed: int
    episodes: int
    eval_every: int
    eval_episodes: int
    alpha: float
    gamma: float
    eps: float
    eps_final: float | None
    eps_decay_episodes: int
    # learning-curve (sample efficiency / convergence speed)
    train_return_auc: float
    train_return_last_mean: float
    # final evaluation
    return_mean: float
    return_std: float
    steps_mean: float
    waste_mean: float
    incompat_mean: float
    made_counts: dict


def _eps_schedule(ep: int, *, eps0: float, eps1: float, decay_n: int) -> float:
    if decay_n <= 0:
        return float(eps0)
    t = min(1.0, (ep - 1) / max(1, decay_n))
    return float((1.0 - t) * eps0 + t * eps1)


def run_episode_tabular(env, agent: TabularAgent, *, train: bool) -> tuple[float, int, dict]:
    _, _ = env.reset()
    key = env.state_key()
    total = 0.0
    steps = 0
    last_info: dict = {}

    a = agent.act_key(key, greedy=not train)
    while True:
        _, reward, terminated, truncated, info = env.step(a)
        total += reward
        steps += 1
        last_info = info
        next_key = env.state_key()

        if train:
            if agent.kind == "sarsa" and not (terminated or truncated):
                na = agent.act_key(next_key, greedy=False)
                agent.update_key(key, a, reward, next_key, terminated, truncated, next_action=na)
                key, a = next_key, na
            else:
                agent.update_key(key, a, reward, next_key, terminated, truncated)
                key = next_key
                a = agent.act_key(key, greedy=False)
        else:
            key = next_key
            a = agent.act_key(key, greedy=True)

        if terminated or truncated:
            break

    return float(total), int(steps), last_info


def run_episode_linear(env, agent: LinearQAgent, *, train: bool) -> tuple[float, int, dict]:
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    last_info: dict = {}

    while True:
        a = agent.act(obs, greedy=not train)
        next_obs, reward, terminated, truncated, info = env.step(a)
        total += reward
        steps += 1
        last_info = info

        if train:
            agent.update(obs, a, reward, next_obs, terminated, truncated)
        obs = next_obs

        if terminated or truncated:
            break

    return float(total), int(steps), last_info


def evaluate(runner, episodes: int) -> dict:
    returns = []
    steps = []
    made = {}
    waste = []
    incompat = []

    for _ in range(int(episodes)):
        ret, st, info = runner(train=False)
        returns.append(float(ret))
        steps.append(int(st))
        if "made" in info:
            made[info["made"]] = made.get(info["made"], 0) + 1
        if "waste" in info:
            waste.append(float(info["waste"]))
        if "incompat" in info:
            incompat.append(float(info["incompat"]))

    def mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "steps_mean": float(np.mean(steps)),
        "made_counts": made,
        "waste_mean": mean(waste),
        "incompat_mean": mean(incompat),
    }


def train_one(
    *,
    algo: str,
    state_mode: str,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    alpha: float,
    gamma: float,
    eps: float,
    eps_final: float | None,
    eps_decay_episodes: int,
    variant: str = "base",
    rounds: int = 1,
    start_budget: float = 10.0,
    allow_debt: bool = False,
    reset_supply_each_round: bool = True,
) -> tuple[RunSummary, dict[str, list[float]]]:
    env = default_env(
        seed=seed,
        config=EnvConfig(
            state_mode=state_mode,  # type: ignore[arg-type]
            variant=variant,  # type: ignore[arg-type]
            rounds=int(rounds),
            start_budget=float(start_budget),
            allow_debt=bool(allow_debt),
            reset_supply_each_round=bool(reset_supply_each_round),
        ),
    )
    # IMPORTANT: seed Gymnasium's internal RNG once; subsequent env.reset() calls will be deterministic.
    env.reset(seed=int(seed))

    eps0 = float(eps)
    eps1 = float(eps_final) if eps_final is not None else float(eps)
    decay_n = int(eps_decay_episodes)

    curves: dict[str, list[float]] = {"ep": [], "return_train": [], "steps_train": []}

    if algo in {"qlearning", "sarsa"}:
        agent = TabularAgent(
            n_actions=env.n_actions,
            alpha=alpha,
            gamma=gamma,
            eps=eps0,
            kind=algo,
            seed=seed,
        )

        def run(*, train: bool):
            return run_episode_tabular(env, agent, train=train)

    elif algo == "linear":
        obs0, _ = env.reset()
        agent = LinearQAgent(
            n_actions=env.n_actions,
            obs_dim=int(obs0.size),
            alpha=min(alpha, 0.1),
            gamma=gamma,
            eps=eps0,
            seed=seed,
        )

        def run(*, train: bool):
            return run_episode_linear(env, agent, train=train)

    else:
        raise ValueError(f"algo non valido: {algo}")

    for ep in range(1, int(episodes) + 1):
        agent.set_eps(_eps_schedule(ep, eps0=eps0, eps1=eps1, decay_n=decay_n))
        ret, st, _ = run(train=True)
        curves["ep"].append(float(ep))
        curves["return_train"].append(float(ret))
        curves["steps_train"].append(float(st))

        # keep it fast: only compute eval stats at the end (or at checkpoints)
        if ep % int(eval_every) == 0:
            _ = evaluate(run, int(eval_episodes))

    # Sample-efficiency style metrics from the training curve
    train_returns = np.array(curves["return_train"], dtype=np.float32)
    train_return_auc = float(np.mean(train_returns)) if train_returns.size > 0 else 0.0
    last_n = int(min(200, train_returns.size))
    train_return_last_mean = float(np.mean(train_returns[-last_n:])) if last_n > 0 else 0.0

    stats = evaluate(run, int(eval_episodes))
    summary = RunSummary(
        algo=algo,
        state_mode=state_mode,
        seed=int(seed),
        episodes=int(episodes),
        eval_every=int(eval_every),
        eval_episodes=int(eval_episodes),
        alpha=float(alpha),
        gamma=float(gamma),
        eps=float(eps),
        eps_final=float(eps_final) if eps_final is not None else None,
        eps_decay_episodes=int(eps_decay_episodes),
        train_return_auc=train_return_auc,
        train_return_last_mean=train_return_last_mean,
        return_mean=float(stats["return_mean"]),
        return_std=float(stats["return_std"]),
        steps_mean=float(stats["steps_mean"]),
        waste_mean=float(stats["waste_mean"]),
        incompat_mean=float(stats["incompat_mean"]),
        made_counts=dict(stats["made_counts"]),
    )
    return summary, curves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results/rlchef")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--eps-final", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=3000)
    # Challenging variant (budget)
    p.add_argument("--variant", choices=["base", "budget"], default=EnvConfig.variant)
    p.add_argument("--rounds", type=int, default=EnvConfig.rounds)
    p.add_argument("--start-budget", type=float, default=EnvConfig.start_budget)
    p.add_argument("--allow-debt", action="store_true")
    p.add_argument("--reset-supply-each-round", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]

    runs: list[RunSummary] = []
    curves_all: dict[str, dict[str, list[float]]] = {}

    # Required comparisons from the project text:
    # - tabular vs function approximation
    # - state representation effect (simple vs mask)
    configs = [
        ("qlearning", "simple"),
        ("qlearning", "mask"),
        ("linear", "simple"),
        ("linear", "mask"),
    ]

    for algo, state_mode in configs:
        for seed in seeds:
            summary, curves = train_one(
                algo=algo,
                state_mode=state_mode,
                episodes=args.episodes,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                seed=seed,
                alpha=args.alpha,
                gamma=args.gamma,
                eps=args.eps,
                eps_final=args.eps_final,
                eps_decay_episodes=args.eps_decay_episodes,
                # env variant flags
                variant=args.variant,
                rounds=args.rounds,
                start_budget=args.start_budget,
                allow_debt=bool(args.allow_debt),
                reset_supply_each_round=bool(args.reset_supply_each_round),
            )
            runs.append(summary)
            curves_all[f"{algo}-{state_mode}-seed{seed}"] = curves
            print(
                f"[{algo:8s} | {state_mode:5s} | seed={seed}] "
                f"R={summary.return_mean:+.3f}±{summary.return_std:.3f} steps={summary.steps_mean:.1f} "
                f"waste={summary.waste_mean:.2f} incompat={summary.incompat_mean:.2f}"
            )

    (outdir / "summaries.json").write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
    (outdir / "curves.json").write_text(json.dumps(curves_all, indent=2), encoding="utf-8")

    # Optional plotting (only if pyplot is available)
    try:
        from matplotlib import pyplot as plt  # type: ignore
    except Exception:
        print("pyplot not available: saved JSON (summaries/curves) but will not generate plots.")
        return

    def plot_group(title: str, keys: list[str], outfile: Path) -> None:
        plt.figure(figsize=(10, 4))
        for k in keys:
            c = curves_all[k]
            plt.plot(c["ep"], c["return_train"], alpha=0.6, label=k)
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel("train return")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

    plot_group(
        "Tabular Q-learning: simple vs mask",
        [k for k in curves_all if k.startswith("qlearning-simple") or k.startswith("qlearning-mask")],
        outdir / "qlearning_simple_vs_mask.png",
    )
    plot_group(
        "Function approximation (linear): simple vs mask",
        [k for k in curves_all if k.startswith("linear-simple") or k.startswith("linear-mask")],
        outdir / "linear_simple_vs_mask.png",
    )


if __name__ == "__main__":
    main()

