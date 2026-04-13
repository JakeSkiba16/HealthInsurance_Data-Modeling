import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
SEED      = 12345
R         = 10_000   # Monte Carlo replications
B         = 499      # Bootstrap resamples (sufficient for coverage studies)
N_LIST    = [25, 50, 100, 250, 500]
ALPHA     = 0.05
TRUE_MEAN = 1.0

rng = np.random.default_rng(SEED)  # modern numpy RNG — faster and seedable

# ─────────────────────────────────────────
# 2. VECTORIZED SAMPLE GENERATORS
# Returns shape (R, n) array — all R samples at once
# ─────────────────────────────────────────

def gen_normal(R, n, rng, mean=1.0, sigma=1.0):
    return rng.normal(loc=mean, scale=sigma, size=(R, n))

def gen_gamma(R, n, rng, mean=1.0, shape=2.0):
    scale = mean / shape
    return rng.gamma(shape=shape, scale=scale, size=(R, n))

def gen_lognormal_moderate(R, n, rng, mean=1.0, sigma=0.75):
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=(R, n))

def gen_lognormal_high(R, n, rng, mean=1.0, sigma=1.5):
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=(R, n))

def gen_mixture(R, n, rng,
                prop_small=0.95,
                mean_small=1.0, sigma_small=0.75,
                mean_large=5.0, sigma_large=2.0):
    """
    Mixture: 95% moderate lognormal + 5% catastrophic lognormal.
    True mean = 0.95 * mean_small + 0.05 * mean_large = 0.95 + 0.25 = 1.20
    To keep true mean = 1.0 exactly, we rescale mean_large so the weighted
    mean equals 1: mean_large = (1.0 - prop_small * mean_small) / (1 - prop_small)
    With mean_small=1: mean_large = (1 - 0.95) / 0.05 = 1.0
    That collapses both components. Instead, fix a meaningful mixture and
    note the true mean is ~1.20, or adjust mean_small downward.
    
    Practical choice: use mean_small=0.8, mean_large=4.0
    True mean = 0.95*0.8 + 0.05*4.0 = 0.76 + 0.20 = 0.96 ≈ 1.0
    Close enough; we store the exact true mean and use it in coverage checks.
    """
    mu_small = np.log(mean_small) - 0.5 * sigma_small**2
    mu_large = np.log(mean_large) - 0.5 * sigma_large**2

    small = rng.lognormal(mean=mu_small, sigma=sigma_small, size=(R, n))
    large = rng.lognormal(mean=mu_large, sigma=sigma_large, size=(R, n))

    # For each of the R*n draws, assign to small or large component
    mask = rng.random(size=(R, n)) < prop_small
    return np.where(mask, small, large)

# Map name → (generator_function, true_mean)
DISTRIBUTIONS = {
    'Normal':           (gen_normal,           1.0),
    'Gamma':            (gen_gamma,            1.0),
    'LognormalModerate':(gen_lognormal_moderate,1.0),
    'LognormalHigh':    (gen_lognormal_high,    1.0),
    'Mixture':          (gen_mixture,           0.95*0.8 + 0.05*4.0),
}

# ─────────────────────────────────────────
# 3. VECTORIZED CONFIDENCE INTERVALS
# samples shape: (R, n)
# Returns lower, upper each shape (R,)
# ─────────────────────────────────────────

def ci_t(samples, alpha=ALPHA):
    n     = samples.shape[1]
    means = samples.mean(axis=1)
    stds  = samples.std(axis=1, ddof=1)
    se    = stds / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    return means - t_crit * se, means + t_crit * se

def ci_z(samples, alpha=ALPHA):
    n     = samples.shape[1]
    means = samples.mean(axis=1)
    stds  = samples.std(axis=1, ddof=1)
    se    = stds / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - alpha/2)
    return means - z_crit * se, means + z_crit * se

def ci_bootstrap(samples, B=B, alpha=ALPHA, seed=0):
    """
    Fully vectorized bootstrap percentile CI.
    samples: (R, n)
    For each of R samples, draw B resamples of size n and compute means.
    Returns lower, upper each shape (R,)
    
    Memory note: intermediate array is shape (R, B) of means — fine for
    R=10000, B=499 (about 40MB of float64).
    """
    R_loc, n = samples.shape
    local_rng = np.random.default_rng(seed)

    # Draw bootstrap indices: shape (R, B, n)
    # To avoid a 3D array that's too large, process in chunks
    # For R=10000, B=499, n=500: 10000*499*500 * 8 bytes = ~20GB — too big
    # So we loop over R in chunks but keep B vectorized

    chunk_size = 500  # tune based on available RAM
    boot_means = np.empty((R_loc, B))

    for start in range(0, R_loc, chunk_size):
        end = min(start + chunk_size, R_loc)
        chunk = samples[start:end]          # shape (chunk, n)
        c = chunk.shape[0]

        # indices: (c, B, n)
        idx = local_rng.integers(0, n, size=(c, B, n))
        # gather: (c, B, n)
        resamples = chunk[np.arange(c)[:, None, None], idx]
        # means: (c, B)
        boot_means[start:end] = resamples.mean(axis=2)

    lower = np.percentile(boot_means, 100 * alpha/2,     axis=1)
    upper = np.percentile(boot_means, 100 * (1 - alpha/2), axis=1)
    return lower, upper

# ─────────────────────────────────────────
# 4. SIMULATION FOR ONE SCENARIO
# ─────────────────────────────────────────

def simulate_scenario(dist_name, n, R=R, B=B, alpha=ALPHA, seed=SEED):
    gen_func, true_mean = DISTRIBUTIONS[dist_name]

    # Use a deterministic per-scenario seed so results are reproducible
    scenario_seed = seed + hash((dist_name, n)) % (2**31)
    local_rng = np.random.default_rng(scenario_seed)

    # Generate all R samples at once — shape (R, n)
    samples = gen_func(R, n, local_rng)

    means = samples.mean(axis=1)   # shape (R,)

    # ── Classical CIs (fully vectorized) ──────────────────────────────
    lo_t, hi_t = ci_t(samples, alpha)
    lo_z, hi_z = ci_z(samples, alpha)

    cov_t = np.mean((lo_t <= true_mean) & (true_mean <= hi_t))
    cov_z = np.mean((lo_z <= true_mean) & (true_mean <= hi_z))
    w_t   = np.mean(hi_t - lo_t)
    w_z   = np.mean(hi_z - lo_z)

    # ── t-test Type I error (vectorized) ──────────────────────────────
    stds    = samples.std(axis=1, ddof=1)
    se      = stds / np.sqrt(n)
    t_stats = (means - true_mean) / se
    t_pvals = 2 * stats.t.sf(np.abs(t_stats), df=n-1)
    type1_t = np.mean(t_pvals < alpha)

    # ── Bootstrap CI ──────────────────────────────────────────────────
    lo_b, hi_b = ci_bootstrap(samples, B=B, alpha=alpha, seed=scenario_seed+1)
    cov_b   = np.mean((lo_b <= true_mean) & (true_mean <= hi_b))
    w_b     = np.mean(hi_b - lo_b)

    # Bootstrap test Type I error: reject if true_mean outside bootstrap CI
    type1_b = np.mean((true_mean < lo_b) | (true_mean > hi_b))

    # ── Shared metrics ─────────────────────────────────────────────────
    bias = np.mean(means) - true_mean
    mse  = np.mean((means - true_mean)**2)

    def mc_se(cov): return np.sqrt(cov * (1 - cov) / R)

    rows = [
        dict(Distribution=dist_name, n=n, Method='t-interval',
             Coverage=cov_t, AvgWidth=w_t, Bias=bias, MSE=mse,
             TypeIError=type1_t, MCStdErrCoverage=mc_se(cov_t)),
        dict(Distribution=dist_name, n=n, Method='z-interval',
             Coverage=cov_z, AvgWidth=w_z, Bias=bias, MSE=mse,
             TypeIError=type1_t, MCStdErrCoverage=mc_se(cov_z)),
        dict(Distribution=dist_name, n=n, Method='bootstrap',
             Coverage=cov_b, AvgWidth=w_b, Bias=bias, MSE=mse,
             TypeIError=type1_b, MCStdErrCoverage=mc_se(cov_b)),
    ]
    return rows

# ─────────────────────────────────────────
# 5. RUN ALL SCENARIOS (PARALLEL)
# ─────────────────────────────────────────

scenarios = [(d, n) for d in DISTRIBUTIONS for n in N_LIST]

print(f"Running {len(scenarios)} scenarios with R={R:,}, B={B} bootstrap resamples...")

all_rows = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_scenario)(dist_name, n)
    for dist_name, n in scenarios
)

# Flatten list of lists
results = [row for scenario_rows in all_rows for row in scenario_rows]
df = pd.DataFrame(results)

# ─────────────────────────────────────────
# 6. SAVE + PRINT
# ─────────────────────────────────────────

df.to_csv("simulation_results.csv", index=False)
print("\n── Results Preview ──────────────────────────")
print(df.to_string(index=False))

# ─────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────

import os
output_dir = "simulation_plots"
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

# ── Plot 1: Distribution Shape Histograms ────────────────────────────

dist_display = {
    'Normal':            ('Normal\n(μ=1, σ=1)',            gen_normal),
    'Gamma':             ('Gamma\n(shape=2, μ=1)',          gen_gamma),
    'LognormalModerate': ('Lognormal Moderate\n(σ=0.75)',   gen_lognormal_moderate),
    'LognormalHigh':     ('Lognormal High\n(σ=1.5)',        gen_lognormal_high),
    'Mixture':           ('Mixture\n(95% mod + 5% cat)',    gen_mixture),
}

hist_rng = np.random.default_rng(999)

fig1, axes1 = plt.subplots(1, 5, figsize=(22, 4))

for ax, (dist_name, (label, gen_func)) in zip(axes1, dist_display.items()):
    sample = gen_func(1, 5000, hist_rng).flatten()  # no clipping — show full tail
    ax.hist(sample, bins=80, color='steelblue', edgecolor='white',
            linewidth=0.3, density=True)
    ax.axvline(sample.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean={sample.mean():.2f}')
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Claim Cost')
    ax.set_ylabel('Density' if ax is axes1[0] else '')
    skew_val = pd.Series(sample).skew()
    ax.text(0.97, 0.95, f'Skew={skew_val:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='darkred')
    ax.legend(fontsize=7)

fig1.suptitle('Simulated Claim Cost Distributions (n=5,000)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot1_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig1)

# ── Plots 2–4: Simulation Results ────────────────────────────────────

def make_plot(y, ylabel, title, filename, hline=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    methods = ['t-interval', 'z-interval', 'bootstrap']
    colors = sns.color_palette("tab10", n_colors=len(DISTRIBUTIONS))

    for ax, method in zip(axes, methods):
        sub = df[df['Method'] == method]
        for (dist_name, color) in zip(DISTRIBUTIONS, colors):
            d = sub[sub['Distribution'] == dist_name].sort_values('n')
            ax.plot(d['n'], d[y], marker='o', label=dist_name, color=color)
        if hline is not None:
            ax.axhline(hline, color='red', linestyle='--', linewidth=1.2, label='Target')
        ax.set_title(method, fontsize=11)
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # critical — prevents figures from stacking

make_plot('Coverage',   'Coverage Probability',  'Coverage Probability vs Sample Size',
          'plot2_coverage.png',  hline=0.95)

make_plot('AvgWidth',   'Average Interval Width', 'Average Interval Width vs Sample Size',
          'plot3_width.png')

make_plot('TypeIError', 'Type I Error Rate',      'Type I Error Rate vs Sample Size',
          'plot4_type1error.png', hline=0.05)

print(f"\nAll plots saved to: {os.path.abspath(output_dir)}/")