import numpy as np
import pandas as pd

np.random.seed(42)

N_ASSETS = 5
N_DAYS = 252 * 5  # 5 years of trading days

dates = pd.bdate_range(start="2021-01-04", periods=N_DAYS)

asset_names = [f"ASSET_{i}" for i in range(N_ASSETS)]

# Annual parameters per asset
annual_means = np.random.uniform(0.02, 0.12, N_ASSETS)
annual_vols = np.random.uniform(0.10, 0.40, N_ASSETS)

# Convert to daily
daily_means = annual_means / 252
daily_vols = annual_vols / np.sqrt(252)

# Generate log returns: r_t ~ N(mu - 0.5*sigma^2, sigma^2)
log_returns = np.random.normal(
    loc=daily_means - 0.5 * daily_vols**2,
    scale=daily_vols,
    size=(N_DAYS, N_ASSETS),
)

# Cumulate to prices starting from 100
prices = 100 * np.exp(np.cumsum(log_returns, axis=0))

df = pd.DataFrame(prices, index=dates, columns=asset_names)
df.index.name = "date"

output_path = __file__.replace("generate_prices.py", "prices.csv")
df.to_csv(output_path)
print(f"Wrote {len(df)} rows x {len(df.columns)} assets to {output_path}")
