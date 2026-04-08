import numpy as np
import pandas as pd


def generate_lognormal(n_assets, n_days, annual_mean_range=(-0.1, 0.25),
                       annual_vol_range=(0.10, 0.80), rng=None):
    """Generate log-normal price paths (geometric Brownian motion)."""
    rng = rng or np.random.default_rng()
    annual_means = rng.uniform(*annual_mean_range, n_assets)
    annual_vols = rng.uniform(*annual_vol_range, n_assets)
    daily_means = annual_means / 252
    daily_vols = annual_vols / np.sqrt(252)

    log_returns = rng.normal(
        loc=daily_means - 0.5 * daily_vols ** 2,
        scale=daily_vols,
        size=(n_days, n_assets),
    )
    return log_returns


def generate_jump_diffusion(n_assets, n_days, annual_mean_range=(-0.05, 0.15),
                            annual_vol_range=(0.15, 0.50),
                            jump_intensity=0.02, jump_mean=-0.03,
                            jump_std=0.05, rng=None):
    """
    Generate Merton jump-diffusion price paths.

    Each day, with probability `jump_intensity`, a jump of size
    N(jump_mean, jump_std^2) is added to the diffusion return.
    This produces fat tails and occasional crashes/spikes.
    """
    rng = rng or np.random.default_rng()
    annual_means = rng.uniform(*annual_mean_range, n_assets)
    annual_vols = rng.uniform(*annual_vol_range, n_assets)
    daily_means = annual_means / 252
    daily_vols = annual_vols / np.sqrt(252)

    # Diffusion component
    diffusion = rng.normal(
        loc=daily_means - 0.5 * daily_vols ** 2,
        scale=daily_vols,
        size=(n_days, n_assets),
    )

    # Jump component
    jump_mask = rng.random((n_days, n_assets)) < jump_intensity
    jump_sizes = rng.normal(loc=jump_mean, scale=jump_std,
                            size=(n_days, n_assets))

    log_returns = diffusion + jump_mask * jump_sizes
    return log_returns


def generate_mean_reverting(n_assets, n_days, long_run_mean=0.0,
                            reversion_speed_range=(0.01, 0.1),
                            annual_vol_range=(0.10, 0.40), rng=None):
    """
    Generate Ornstein-Uhlenbeck (mean-reverting) log-return paths.

    Useful for simulating assets like pairs-traded spreads, interest
    rates, or commodities that tend to revert to a long-run level.
    """
    rng = rng or np.random.default_rng()
    kappa = rng.uniform(*reversion_speed_range, n_assets)
    annual_vols = rng.uniform(*annual_vol_range, n_assets)
    daily_vols = annual_vols / np.sqrt(252)

    log_returns = np.zeros((n_days, n_assets))
    level = np.full(n_assets, long_run_mean)

    for t in range(n_days):
        noise = rng.normal(scale=daily_vols, size=n_assets)
        level = level + kappa * (long_run_mean - level) + noise
        log_returns[t] = level

    return log_returns


def generate_regime_switching(n_assets, n_days, n_regimes=2,
                              transition_prob=0.005,
                              annual_vol_range=(0.10, 0.80), rng=None):
    """
    Generate returns that switch between bull/bear regimes.

    Each regime has its own mean and volatility. Transitions happen
    with probability `transition_prob` per day (Markov chain).
    """
    rng = rng or np.random.default_rng()
    annual_vols = rng.uniform(*annual_vol_range, n_assets)
    daily_vols = annual_vols / np.sqrt(252)

    # Define regime parameters: regime 0 = bull, regime 1 = bear, etc.
    regime_means = np.linspace(0.10, -0.15, n_regimes) / 252
    regime_vol_multipliers = np.linspace(0.7, 2.0, n_regimes)

    log_returns = np.zeros((n_days, n_assets))
    regime = 0

    for t in range(n_days):
        if rng.random() < transition_prob:
            regime = rng.integers(0, n_regimes)
        mu = regime_means[regime]
        vol = daily_vols * regime_vol_multipliers[regime]
        log_returns[t] = rng.normal(loc=mu - 0.5 * vol ** 2, scale=vol)

    return log_returns


def generate_mixed_universe(config, n_days=252 * 10, seed=42):
    """
    Generate a mixed-asset universe with different return dynamics.

    Parameters
    ----------
    config : list of dict
        Each dict specifies a group of assets:
            {'type': 'lognormal' | 'jump_diffusion' | 'mean_reverting' | 'regime_switching',
             'n_assets': int,
             **kwargs}  # extra kwargs passed to the generator
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Price DataFrame indexed by business dates.

    Example
    -------
    >>> df = generate_mixed_universe([
    ...     {'type': 'lognormal', 'n_assets': 20},
    ...     {'type': 'jump_diffusion', 'n_assets': 10, 'jump_intensity': 0.03},
    ...     {'type': 'mean_reverting', 'n_assets': 5},
    ...     {'type': 'regime_switching', 'n_assets': 5},
    ... ])
    """
    generators = {
        'lognormal': generate_lognormal,
        'jump_diffusion': generate_jump_diffusion,
        'mean_reverting': generate_mean_reverting,
        'regime_switching': generate_regime_switching,
    }

    rng = np.random.default_rng(seed)
    all_log_returns = []
    asset_names = []
    asset_idx = 0

    for group in config:
        group = group.copy()
        asset_type = group.pop('type')
        n_assets = group.pop('n_assets')

        gen_fn = generators[asset_type]
        log_returns = gen_fn(n_assets=n_assets, n_days=n_days, rng=rng,
                             **group)
        all_log_returns.append(log_returns)

        prefix = asset_type.upper()[:4]
        for i in range(n_assets):
            asset_names.append(f"{prefix}_{asset_idx}")
            asset_idx += 1

    log_returns = np.hstack(all_log_returns)
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))

    dates = pd.bdate_range(start="2021-01-04", periods=n_days)
    df = pd.DataFrame(prices, index=dates, columns=asset_names)
    df.index.name = "date"
    return df


if __name__ == "__main__":
    config = [
        {'type': 'lognormal', 'n_assets': 20},
        {'type': 'jump_diffusion', 'n_assets': 15, 'jump_intensity': 0.02},
        {'type': 'mean_reverting', 'n_assets': 10},
        {'type': 'regime_switching', 'n_assets': 5},
    ]

    df = generate_mixed_universe(config)

    output_path = __file__.replace("generate_prices.py", "prices.csv")
    df.to_csv(output_path)
    print(f"Wrote {len(df)} rows x {len(df.columns)} assets to {output_path}")
    print(f"\nAsset breakdown:")
    for group in config:
        print(f"  {group['n_assets']:>3} x {group['type']}")
