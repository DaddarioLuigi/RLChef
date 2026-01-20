from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from rlchef.agents import LinearQAgent, TabularAgent
from rlchef.env import ChefGridEnv, EnvConfig, default_env


@dataclass
class TrainConfig:
    episodes: int = 5000
    eval_every: int = 500
    eval_episodes: int = 200
    seed: int = 0

    alpha: float = 0.1
    gamma: float = 0.99
    eps: float = 0.1


def run_episode_tabular(env: ChefGridEnv, agent: TabularAgent, *, train: bool) -> tuple[float, int, dict]:
    obs, _ = env.reset()
    key = env.state_key()
    total = 0.0
    steps = 0
    last_info: dict = {}

    a = agent.act_key(key, greedy=not train)
    while True:
        next_obs, reward, terminated, truncated, info = env.step(a)
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

    return total, steps, last_info


def run_episode_linear(env: ChefGridEnv, agent: LinearQAgent, *, train: bool) -> tuple[float, int, dict]:
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

    return total, steps, last_info


def evaluate(env: ChefGridEnv, runner, episodes: int) -> dict:
    returns = []
    steps = []
    made = {}
    waste = []
    incompat = []

    for _ in range(episodes):
        ret, st, info = runner(train=False)
        returns.append(ret)
        steps.append(st)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["qlearning", "sarsa", "linear"], required=True)
    p.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--alpha", type=float, default=TrainConfig.alpha)
    p.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    p.add_argument("--eps", type=float, default=TrainConfig.eps)
    p.add_argument("--eps-final", type=float, default=None)
    p.add_argument("--eps-decay-episodes", type=int, default=0)
    p.add_argument("--state-mode", choices=["simple", "mask"], default=EnvConfig.state_mode)
    p.add_argument("--max-steps", type=int, default=EnvConfig.max_steps)
    p.add_argument("--fail-penalty", type=float, default=EnvConfig.fail_penalty)
    p.add_argument("--empty-cook-extra-penalty", type=float, default=EnvConfig.empty_cook_extra_penalty)
    # Challenging variant (budget)
    p.add_argument("--variant", choices=["base", "budget"], default=EnvConfig.variant)
    p.add_argument("--rounds", type=int, default=EnvConfig.rounds)
    p.add_argument("--start-budget", type=float, default=EnvConfig.start_budget)
    p.add_argument("--allow-debt", action="store_true")
    p.add_argument("--reset-supply-each-round", action="store_true")
    args = p.parse_args()

    env_cfg = EnvConfig(
        max_steps=args.max_steps,
        fail_penalty=args.fail_penalty,
        empty_cook_extra_penalty=args.empty_cook_extra_penalty,
        state_mode=args.state_mode,
        variant=args.variant,
        rounds=args.rounds,
        start_budget=args.start_budget,
        allow_debt=bool(args.allow_debt),
        reset_supply_each_round=bool(args.reset_supply_each_round),
    )
    env = default_env(seed=args.seed, config=env_cfg)
    # IMPORTANT: seed Gymnasium's internal RNG once; subsequent env.reset() calls will be deterministic.
    env.reset(seed=args.seed)

    if args.algo in {"qlearning", "sarsa"}:
        agent = TabularAgent(
            n_actions=env.n_actions,
            alpha=args.alpha,
            gamma=args.gamma,
            eps=args.eps,
            kind=args.algo,
            seed=args.seed,
        )
        def run(*, train: bool):
            return run_episode_tabular(env, agent, train=train)
    else:
        obs0, _ = env.reset()
        agent = LinearQAgent(
            n_actions=env.n_actions,
            obs_dim=int(obs0.size),
            alpha=min(args.alpha, 0.1),
            gamma=args.gamma,
            eps=args.eps,
            seed=args.seed,
        )
        def run(*, train: bool):
            return run_episode_linear(env, agent, train=train)

    eps0 = float(args.eps)
    eps1 = float(args.eps_final) if args.eps_final is not None else float(args.eps)
    decay_n = int(args.eps_decay_episodes)

    for ep in range(1, args.episodes + 1):
        if decay_n > 0:
            t = min(1.0, (ep - 1) / max(1, decay_n))
            eps = (1.0 - t) * eps0 + t * eps1
            agent.set_eps(eps)
        run(train=True)
        if ep % args.eval_every == 0 or ep == args.episodes:
            stats = evaluate(env, run, args.eval_episodes)
            made_top = sorted(stats["made_counts"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(
                f"ep={ep:6d} "
                f"R={stats['return_mean']:+.3f}±{stats['return_std']:.3f} "
                f"steps={stats['steps_mean']:.1f} "
                f"waste={stats['waste_mean']:.2f} "
                f"incompat={stats['incompat_mean']:.2f} "
                f"made_top={made_top}"
            )


if __name__ == "__main__":
    main()

