# evoport

Evolutionary multi-objective portfolio optimisation in Python.

**evoport** constructs portfolios by evolving a population of weight vectors over
multiple generations. At each step, portfolios are evaluated on competing risk and
return objectives, non-dominated solutions are selected via
epsilon-dominance, and new candidates are bred through Gaussian perturbation with
projection back onto the feasible set.

## Quick Start

```bash
pip install -e .
```

```python
from utils.populations.implementations import LongOnlyBC
from utils.loss.sharpe import Sharpe
from utils.loss.cvar import CVaR
from utils.loss.template import ListLoss
from utils.domination import epsilon

population = LongOnlyBC(problem_dimension=5, population_size=50_000)
losses = ListLoss([Sharpe('Sharpe'), CVaR('CVaR')])

# Evaluate and select
results = losses.evaluate(population.initial_weights, data_package)
frontier = epsilon.epsilon_dominant(df=results, ...)
```

See [`examples/populations.ipynb`](examples/populations.ipynb) for a full walkthrough.

## Features

| Component | Description |
|---|---|
| **Populations** | `LongOnlyBC` (budget-constrained, fully invested) and `LongOnlyBR` (budget-relaxed, partial cash) |
| **Objectives** | Sharpe, Sortino, Mean Return, Max Drawdown, CVaR, VaR, Win Rate, Win/Loss |
| **Domination** | Pareto, Epsilon, Cone, Lorenz |
| **Breeding** | Gaussian perturbation with automatic projection to feasibility |

## How It Works

### 1. Population Sampling

Portfolios are sampled uniformly over the feasible weight simplex. Two constraint
types are supported:

- **Budget-Constrained (BC):** weights are non-negative and sum to exactly 1.
- **Budget-Relaxed (BR):** weights are non-negative and sum to at most 1, allowing a cash position.

![Weight Space](images/weight_space.png)

### 2. Multi-Objective Evaluation

Each portfolio is evaluated on multiple objectives simultaneously. Here we optimise
the annualised Sharpe ratio against CVaR at 5%. The initial population spans the
full objective space, and epsilon-dominance identifies the non-dominated frontier.

![Initial Frontier](images/initial_frontier.png)

### 3. Evolutionary Selection

Over successive generations, dominated portfolios are culled and replaced by
offspring of frontier members. The frontier compresses toward the Pareto-optimal
trade-off curve.

![Frontier Evolution](images/frontier_evolution.png)

### 4. Objective Distributions

The population's objective distributions shift rightward (better) with each
generation, confirming systematic improvement.

![Objective Distributions](images/objective_distributions.png)

### 5. Convergence

Median objectives improve monotonically. Selection pressure (frontier fraction)
stabilises as the population concentrates near optimality.

![Convergence](images/convergence.png)

### 6. Before and After

The contrast between the initial random population and the evolved population in
objective space.

![Initial vs Final](images/initial_vs_final.png)

### 7. Portfolio Composition

Top-ranked frontier portfolios concentrate weight in the highest risk-adjusted-return
assets, with smooth variation along the frontier.

![Weight Heatmap](images/weight_heatmap.png)

## Project Structure

```
evoport/
  data/              # Price data and generation scripts
  examples/          # Jupyter notebooks
  images/            # Generated figures
  utils/
    domination/      # Pareto, epsilon, cone, Lorenz dominance
    enums/           # Data enums
    loss/            # Objective functions (Sharpe, CVaR, etc.)
    populations/     # Population sampling and breeding
```

## Dependencies

- NumPy, Pandas, SciPy, Matplotlib, Seaborn
- CVXPY (for convex baselines)

## License

MIT
