from __future__ import annotations

import argparse

from rlchef.agents import LinearQAgent, TabularAgent
from rlchef.env import EnvConfig, default_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["qlearning", "sarsa", "linear"], required=True)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--eps-final", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=2000)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--state-mode", choices=["simple", "mask"], default=EnvConfig.state_mode)
    # Challenging variant (budget)
    p.add_argument("--variant", choices=["base", "budget"], default=EnvConfig.variant)
    p.add_argument("--rounds", type=int, default=EnvConfig.rounds)
    p.add_argument("--start-budget", type=float, default=EnvConfig.start_budget)
    p.add_argument("--allow-debt", action="store_true")
    p.add_argument("--reset-supply-each-round", action="store_true")
    args = p.parse_args()

    env = default_env(
        seed=args.seed,
        config=EnvConfig(
            state_mode=args.state_mode,
            variant=args.variant,
            rounds=args.rounds,
            start_budget=args.start_budget,
            allow_debt=bool(args.allow_debt),
            reset_supply_each_round=bool(args.reset_supply_each_round),
        ),
    )
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
        for ep in range(1, args.episodes + 1):
            if args.eps_decay_episodes > 0:
                t = min(1.0, (ep - 1) / max(1, args.eps_decay_episodes))
                agent.set_eps((1.0 - t) * args.eps + t * args.eps_final)

            env.reset()
            key = env.state_key()
            a = agent.act_key(key, greedy=False)
            while True:
                _, reward, terminated, truncated, info = env.step(a)
                nk = env.state_key()
                if agent.kind == "sarsa" and not (terminated or truncated):
                    na = agent.act_key(nk, greedy=False)
                    agent.update_key(key, a, reward, nk, terminated, truncated, next_action=na)
                    key, a = nk, na
                else:
                    agent.update_key(key, a, reward, nk, terminated, truncated)
                    key = nk
                    a = agent.act_key(key, greedy=False)
                if terminated or truncated:
                    break
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
        for ep in range(1, args.episodes + 1):
            if args.eps_decay_episodes > 0:
                t = min(1.0, (ep - 1) / max(1, args.eps_decay_episodes))
                agent.set_eps((1.0 - t) * args.eps + t * args.eps_final)

            obs, _ = env.reset()
            while True:
                a = agent.act(obs, greedy=False)
                next_obs, reward, terminated, truncated, info = env.step(a)
                agent.update(obs, a, reward, next_obs, terminated, truncated)
                obs = next_obs
                if terminated or truncated:
                    break

    agent.set_eps(0.0)
    obs, _ = env.reset()
    alive = env.render(last_reward=0.0, made=None, fps=args.fps)
    if not alive:
        env.close()
        return

    while True:
        if args.algo in {"qlearning", "sarsa"}:
            a = agent.act_key(env.state_key(), greedy=True)
        else:
            a = agent.act(obs, greedy=True)

        next_obs, reward, terminated, truncated, info = env.step(a)
        obs = next_obs
        made = info.get("made") if terminated else None
        alive = env.render(last_reward=reward, made=made, fps=args.fps)
        if not alive:
            break
        if terminated or truncated:
            while alive:
                alive = env.render(last_reward=reward, made=info.get("made"), fps=args.fps)
            break

    env.close()


if __name__ == "__main__":
    main()

