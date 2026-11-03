"""
OEX: model-implied volatility surface vs. market (historical-vol) surface,
before and after COVID.   ** VERSION WITH A 1-YEAR (252 trading-day) MATURITY **

Change versus OEX_surface_comparison.py: the 1-year maturity is added to the
whole analysis (tuning targets, surface grid, tables, figures).  The data files
contain historical vol only up to 180 days, so the 252-day HV is computed here
from the price series: sample std (ddof=1) of daily log returns x sqrt(252).
This formula reproduces the vendor's 10d, 60d and 180d columns exactly.
Because the series starts 2016-01-04, 252d HV exists only from 2017-01-03.  To keep
the term structure consistent, every tenor is averaged over the same dates (rows
where all nine tenors exist), so the pre-COVID period is 2017-01-03 to 2019-12-31
(the original script averaged the 8 shorter tenors from 2016-01-04).

Methodology (Part A/B of SurfaceVolatility.py is imported unchanged):
  - perturbation expansion coefficients  -> sv.compute_coefficients
  - Black-Scholes put (variance input)   -> sv.BS_put
  - implied-vol inversion (Brent)        -> sv.implied_vol_put

Usage:  python OEX_surface_comparison.py [DATA_DIR] [OUT_DIR]
  DATA_DIR : folder holding the three .npz files   (default: this script's folder)
  OUT_DIR  : folder for figures / CSVs             (default: ./output)
SurfaceVolatility.py must sit next to this script.
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import SurfaceVolatility as sv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else HERE
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ settings
SPLIT = "fine"        # "fine" (5 regimes) or "coarse" (pre / post COVID)
PERIOD_SETS = {
    "coarse": {
        "Pre-COVID (2016-2019)":  ("2016-01-01", "2019-12-31"),
        "Post-COVID (2020-2026)": ("2020-01-01", "2026-09-30"),
    },
    "fine": {
        "Pre-COVID (2016-2019)":      ("2016-01-01", "2019-12-31"),
        "COVID shock (2020)":         ("2020-01-01", "2020-12-31"),
        "Recovery (2021)":            ("2021-01-01", "2021-12-31"),
        "Tightening bear (2022)":     ("2022-01-01", "2022-12-31"),
        "Normalisation (2023-2026)":  ("2023-01-01", "2026-09-30"),
    },
}
PERIODS = PERIOD_SETS[SPLIT]
# Guessed (not fitted) parameters: HV term structure cannot identify them.
# lam = vol-of-vol, rho = spot/vol correlation (same guess for every period)
FIXED = {name: dict(lam=0.8, rho=-0.7) for name in PERIODS}
H = 0.01                       # finite-difference step (script default is 0.5)
TRADING_DAYS = 252
HV_TENORS = [10, 20, 30, 60, 90, 120, 150, 180, 252]   # 252 trading days = 1 year
MONEYNESS = np.linspace(0.9, 1.1, 41)          # K / S0 (+-10%; expansion is unreliable further out)
WING_C = 2.5   # keep only |ln(K/S0)| <= WING_C * sigma * sqrt(T); beyond that the
               # expansion price error swamps the (tiny) option value and IV is meaningless

# ------------------------------------------------------------------ data
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame(d["data"], columns=d["columns"], index=d["index"])

def find(prefix):
    return sorted(os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                  if f.startswith(prefix) and f.endswith(".npz"))

hv = pd.concat([load_npz(p) for p in find("ivol_OEX_historical_volatility")],
               ignore_index=True)
hv["date"] = pd.to_datetime(hv["date"])
hv = hv.sort_values("date").reset_index(drop=True)
hv["price"] = hv["price"].astype(float)
_lr = np.log(hv["price"]).diff()
for _n in HV_TENORS:
    _c = f"{_n}d HV"
    if _c in hv.columns:
        hv[_c] = hv[_c].astype(float)                       # vendor column
    else:                                                   # not supplied -> compute from prices
        hv[_c] = _lr.rolling(_n).std(ddof=1) * np.sqrt(TRADING_DAYS)
hv_cols = [f"{n}d HV" for n in HV_TENORS]
hv = hv[["date", "price"] + hv_cols].set_index("date")

rates = load_npz(find("ivol_interest_rates_USD")[0])
rates["t_date"] = pd.to_datetime(rates["t_date"])
rates["period"] = rates["period"].astype(int)
rates["rate"] = rates["rate"].astype(float) / 100.0      # percent -> decimal

# T grid = 9 HV tenors + midpoints (17 maturities, 10 days to 1 year); ATM points are the 9 tenors
T_tenor = np.array(HV_TENORS) / TRADING_DAYS
T_grid = np.sort(np.concatenate([T_tenor, (T_tenor[:-1] + T_tenor[1:]) / 2]))
tenor_idx = [int(np.argmin(abs(T_grid - t))) for t in T_tenor]

# ------------------------------------------------------------------ model
def price_expansion_h(x, psi, K, r, T, coeffs, h=H):
    """Same expansion as sv.price_expansion, but with a configurable step h."""
    a0, a1, a2, b0, b2 = coeffs
    P = lambda xx: sv.BS_put(xx, psi, K, r, T)
    P0 = P(x)
    d1 = (P(x + h) - P(x - h)) / (2 * h)
    d2 = (P(x + h) - 2 * P0 + P(x - h)) / h**2
    d3 = (P(x + 2*h) - 2*P(x + h) + 2*P(x - h) - P(x - 2*h)) / (2 * h**3)
    d4 = (P(x + 2*h) - 4*P(x + h) + 6*P0 - 4*P(x - h) + P(x - 2*h)) / h**4
    return P0 + a0*d1 + a1*d2 + a2*d3 + b0*d2 + b2*d4

def model_surface(S0, r_of_T, V0, kappa, theta, lam, rho, Ts, Ks_rel):
    """IV[i, j] for maturity Ts[i], strike Ks_rel[j]*S0.
    Coefficients depend on T only, so they are computed once per maturity."""
    x0 = np.log(S0)
    iv = np.full((len(Ts), len(Ks_rel)), np.nan)
    for i, T in enumerate(Ts):
        psi, a0, a1, a2, b0, b2 = sv.compute_coefficients(V0, kappa, theta, lam, rho, T)
        r = r_of_T[i]
        for j, m in enumerate(Ks_rel):
            K = m * S0
            p = price_expansion_h(x0, psi, K, r, T, (a0, a1, a2, b0, b2))
            iv[i, j] = sv.implied_vol_put(p, S0, K, r, T)
    return iv

def atm_curve(S0, r_tenor, V0, kappa, theta, lam, rho):
    x0 = np.log(S0)
    out = []
    for T, r in zip(T_tenor, r_tenor):
        psi, a0, a1, a2, b0, b2 = sv.compute_coefficients(V0, kappa, theta, lam, rho, T)
        p = price_expansion_h(x0, psi, S0, r, T, (a0, a1, a2, b0, b2))
        out.append(sv.implied_vol_put(p, S0, S0, r, T))
    return np.array(out)

# ------------------------------------------------------------------ run
results, summary_rows, term_rows = {}, [], []
for name, (d0, d1) in PERIODS.items():
    sub = hv.loc[d0:d1].dropna(subset=hv_cols)           # common sample: every tenor averaged over the same dates
    mkt = sub[hv_cols].mean().values                      # market (HV) term structure
    S0 = float(sub["price"].iloc[-1])                     # period-end spot
    curve = (rates[(rates.t_date >= d0) & (rates.t_date <= d1)]
             .groupby("period")["rate"].mean())
    r_of = lambda T: np.interp(T * 365, curve.index.values, curve.values)
    r_grid, r_tenor = r_of(T_grid), r_of(T_tenor)
    lam, rho = FIXED[name]["lam"], FIXED[name]["rho"]

    # Light tuning of V0, theta, kappa so the model ATM IV follows the HV term
    # structure. theta is kept within [1.0, 1.6] x the long-end HV so the fit
    # cannot drift to implausible long-run levels.
    lo = float(mkt[-1])
    def resid(p):
        V0, theta, kappa = p
        m = atm_curve(S0, r_tenor, V0, kappa, theta, lam, rho)
        return np.where(np.isnan(m), 1.0, m - mkt)
    fit = least_squares(resid, x0=[mkt[0], 1.2 * lo, 2.0],
                        bounds=([0.02, lo, 0.1], [0.8, 1.6 * lo, 10.0]))
    V0, theta, kappa = fit.x

    iv = model_surface(S0, r_grid, V0, kappa, theta, lam, rho, T_grid, MONEYNESS)
    sig_T = np.interp(T_grid, T_tenor, mkt)
    wing = np.abs(np.log(MONEYNESS))[None, :] > WING_C * sig_T[:, None] * np.sqrt(T_grid)[:, None]
    iv_raw = iv.copy()
    iv[wing] = np.nan
    atm_model = iv[tenor_idx, int(np.argmin(abs(MONEYNESS - 1.0)))]
    results[name] = dict(iv=iv, mkt=mkt, S0=S0, params=(V0, kappa, theta, lam, rho))

    for t, mk, md in zip(HV_TENORS, mkt, atm_model):
        term_rows.append(dict(Period=name, Tenor_days=t, T_years=t / TRADING_DAYS,
                              Market_HV=mk, Model_ATM_IV=md, Diff=md - mk))
    j9, j11 = int(np.argmin(abs(MONEYNESS - .9))), int(np.argmin(abs(MONEYNESS - 1.1)))
    i3m = int(np.argmin(abs(T_grid - 0.25)))
    i1y = int(np.argmin(abs(T_grid - 1.0)))
    k180 = HV_TENORS.index(180)
    summary_rows.append(dict(
        Period=name, S0=S0, Market_HV_10d=mkt[0], Market_HV_180d=mkt[k180], Market_HV_1y=mkt[-1],
        Model_ATM_IV_1y=atm_model[-1],
        Market_slope_180_minus_10=mkt[k180] - mkt[0], Market_slope_1y_minus_10=mkt[-1] - mkt[0], V0=V0, kappa=kappa, theta=theta, lam=lam, rho=rho,
        RMSE_ATM_vs_HV=float(np.sqrt(np.mean((atm_model - mkt) ** 2))),
        Skew_3M_IV90_minus_IV110=iv[i3m, j9] - iv[i3m, j11],
        Skew_1Y_IV90_minus_IV110=iv[i1y, j9] - iv[i1y, j11],
        Share_of_grid_shown=float(np.isfinite(iv).mean()),
        Share_inversion_failed_inside_window=float(np.isnan(iv_raw[~wing]).mean())))

    # CSV of the model surface
    rows = [dict(Period=name, Maturity_Years=T_grid[i], Moneyness=MONEYNESS[j],
                 Strike=MONEYNESS[j] * S0, Model_IV=iv[i, j],
                 Market_HV=float(np.interp(T_grid[i], T_tenor, mkt)))
            for i in range(len(T_grid)) for j in range(len(MONEYNESS))]
    tag = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"OEX_surface_{tag}.csv"), index=False)

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "OEX_calibration_summary.csv"), index=False)
pd.DataFrame(term_rows).to_csv(os.path.join(OUT_DIR, "OEX_term_structure_comparison.csv"), index=False)

# ------------------------------------------------------------------ plots
names = list(results)
Lm, Tm = np.meshgrid(np.log(MONEYNESS), T_grid)

# (1) 3-D: model surface vs market (HV) surface, one panel per period
n = len(names)
ncols = min(n, 3); nrows = int(np.ceil(n / ncols))
zmin = min(np.nanmin(r["iv"]) for r in results.values()) - 0.01
zmax = max(np.nanmax(r["iv"]) for r in results.values()) + 0.01
fig = plt.figure(figsize=(6.2 * ncols, 5.6 * nrows))
for k, name in enumerate(names):
    r = results[name]
    ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
    ax.plot_surface(Lm, Tm, np.ma.masked_invalid(r["iv"]), cmap="viridis", alpha=0.9, edgecolor="none")
    mk_grid = np.tile(np.interp(T_grid, T_tenor, r["mkt"])[:, None], (1, len(MONEYNESS)))
    mk_grid = np.where(np.isfinite(r["iv"]), mk_grid, np.nan)
    ax.plot_surface(Lm, Tm, np.ma.masked_invalid(mk_grid), color="tab:red", alpha=0.45, edgecolor="none")
    ax.set_title(name, fontsize=11); ax.set_zlim(zmin, zmax)
    ax.set_xlabel("ln(K/S0)"); ax.set_ylabel("T (yrs)"); ax.set_zlabel("Vol")
    ax.view_init(elev=22, azim=-60)
fig.suptitle("OEX model-implied surface (coloured) vs market historical-vol surface (red, flat in strike)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "OEX_surfaces_model_vs_market.png"), dpi=150)
plt.close(fig)

# (2) 2-D comparison: term structure + 3-month smile
colors = plt.cm.tab10(np.arange(n))
fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
ax = axes[0]
for name, c in zip(names, colors):
    r = results[name]
    ax.plot(HV_TENORS, r["mkt"], "o-", color=c, label=f"{name} market (HV)")
    ax.plot(HV_TENORS, [x["Model_ATM_IV"] for x in term_rows if x["Period"] == name],
            "s--", color=c, ms=4, alpha=0.9)
ax.set_xlabel("Tenor (trading days)"); ax.set_ylabel("Volatility")
ax.set_title("Term structure: market HV (solid) vs model ATM IV (dashed)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axes[1]
i3m = int(np.argmin(abs(T_grid - 0.25)))
for name, c in zip(names, colors):
    r = results[name]
    ax.plot(MONEYNESS, r["iv"][i3m], color=c, label=name)
    ax.axhline(np.interp(T_grid[i3m], T_tenor, r["mkt"]), ls=":", lw=1, color=c)
ax.set_xlabel("Moneyness K/S0"); ax.set_ylabel("Implied vol")
ax.set_title(f"Model smile at T = {T_grid[i3m]:.2f}y (dotted = market HV)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axes[2]
i1y = int(np.argmin(abs(T_grid - 1.0)))
for name, c in zip(names, colors):
    r = results[name]
    ax.plot(MONEYNESS, r["iv"][i1y], color=c, label=name)
    ax.axhline(np.interp(T_grid[i1y], T_tenor, r["mkt"]), ls=":", lw=1, color=c)
ax.set_xlabel("Moneyness K/S0"); ax.set_ylabel("Implied vol")
ax.set_title(f"Model smile at T = {T_grid[i1y]:.2f}y (dotted = market HV)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "OEX_term_structure_and_smiles.png"), dpi=150)
plt.close(fig)

print(pd.DataFrame(summary_rows).round(4).to_string(index=False))
print()
print(pd.DataFrame(term_rows).round(4).to_string(index=False))
print("\nSaved to", OUT_DIR)
