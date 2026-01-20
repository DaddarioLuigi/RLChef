# RL Chef (Gridworld)

MDP environment on a 5x5 grid: the agent (chef) moves between markets, collects ingredients, and decides when to stop and try to cook a dish. The objective is to maximize dish value while penalizing waste and incompatible ingredient combinations.

The environment is implemented as a **`gymnasium.Env`** (with `action_space`, `observation_space`, `reset(seed=...)`, and `step()` following the Gymnasium API).

## Alignment with the project description (Chef)

The tasks required by the PDF are covered as follows:

- **Model the environment as an MDP**: `rlchef/env.py` defines state/actions/reward and implements the Gymnasium environment.
- **Two RL agents**:
  - tabular: `TabularAgent` (Q-learning or SARSA) in `rlchef/agents.py`
  - function approximation: `LinearQAgent` (linear Q approximation) in `rlchef/agents.py`
- **Compare learning performance / policy efficiency**: `rlchef/experiments.py` saves learning curves and summaries (multi-seed).
- **Analyze state representation effects**: `state_mode=simple` (lossy) vs `state_mode=mask` (Markov) in `rlchef/env.py`.

### Important note on Markovity and state representation

- The environment uses a **fixed grid** by default (`grid_mode="fixed"`) to ensure reproducible dynamics.
- With `state_mode="mask"`, the tabular state key includes a visited-cells mask: this is the **Markov** version (given a fixed layout).
- With `state_mode="simple"`, the tabular state key is intentionally incomplete (lossy): it is used to study how a poor representation slows down or prevents convergence.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS notes (pygame)

If `pip install -r requirements.txt` tries to build and fails with `sdl2-config: command not found`, use one of the following:

```bash
# Option A (recommended): pygame-ce (already in requirements.txt)
python -m pip install -r requirements.txt

# Option B: conda (includes SDL)
# conda install -c conda-forge pygame

# Option C: Homebrew + build
# brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf pkg-config
# python -m pip install pygame
```

## Run

Training and comparison between:
- a tabular agent (Q-learning or SARSA)
- a function-approximation agent (linear)

```bash
python -m rlchef.train --algo qlearning --episodes 5000
python -m rlchef.train --algo sarsa --episodes 5000
python -m rlchef.train --algo linear --episodes 5000
```

## ε-decay (recommended)

```bash
python -m rlchef.train --algo linear --episodes 5000 --eps 0.3 --eps-final 0.05 --eps-decay-episodes 3000
```

## State: simple vs mask

To study the impact of state representation on tabular methods:

```bash
python -m rlchef.train --algo qlearning --state-mode simple --episodes 5000
python -m rlchef.train --algo qlearning --state-mode mask --episodes 5000
```

## Experiments (reproducible results)

Runs the 4 required combinations (tabular/linear × simple/mask) across multiple seeds and saves:
- `summaries.json` (final metrics)
- `curves.json` (per-episode learning curves)
- In `summaries.json` we also store simple **sample-efficiency / convergence** indicators computed from the training curve:
  - `train_return_auc`: mean training return over all episodes (area-under-curve proxy)
  - `train_return_last_mean`: mean training return over the last episodes (stability / late performance)
- `.png` plots if **pyplot** (from `matplotlib`) is available

```bash
python -m rlchef.experiments --episodes 5000 --seeds 0,1,2 --outdir results/rlchef
```

## Challenging variant (budget / costs / multi-round wealth)

Enable the challenging variant by setting `--variant budget`. In this mode:
- each ingredient has a **cost** (paid when picked up)
- cooking yields **revenue** (added to the chef budget)
- the episode lasts for `--rounds` cooking rounds (unless truncated by `--max-steps`)

Examples:

```bash
python -m rlchef.train --algo qlearning --variant budget --rounds 5 --start-budget 10 --state-mode mask --episodes 5000
python -m rlchef.train --algo linear --variant budget --rounds 5 --start-budget 10 --episodes 5000
python -m rlchef.visualize --algo qlearning --variant budget --rounds 5 --start-budget 10 --state-mode mask --episodes 3000
```

## Useful parameters

```bash
python -m rlchef.train --help
```

## Visualization (pygame)

Train and then render 1 greedy episode in a window (close the window to exit):

```bash
python -m rlchef.visualize --algo qlearning --episodes 3000 --fps 12
python -m rlchef.visualize --algo linear --episodes 3000 --fps 12
```

