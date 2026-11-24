import os
import numpy as np
from math import exp, log, sqrt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# ============================================================
# PART (A): STOCHASTIC VOLATILITY MODEL AND OPTION PRICING
# ------------------------------------------------------------
# This part implements Stephen Diehl’s stochastic volatility
# model and perturbation-based option pricing expansion.
#
# SOURCE:
# https://www.stephendiehl.com/posts/volatility_surface/
#
# OUTPUT:
# Option prices P_model(K, T)
# ============================================================


# ============================================================
# REAL MARKET DATA (EMBEDDED FROM XLSX FILES)
# ============================================================
#Spot_FX.xlsx
spot_rates = {
    #"USDEUR": 1.1568,
    "USDJPY": 152.54,
    "USDAUD": 1.5258
    #"USDSGD": xxxxxx,
}
#Rates.xlsx (foreign_rate, dmestic_rate)
"""usd_eur_rates = {
    1/12:(0.0419,0.01962395652),
    3/12:(0.0402,0.02034434783),
    6/12:(0.0381,0.02034434783),
    1.0 :(0.0360,0.02034434783)
}"""
#Rates.xlsx (foreign_rate, dmestic_rate)
usd_jpy_rates = {
    1/12:(0.0419,0.00587696667),
    3/12:(0.0402,0.0080909),
    6/12:(0.0381,0.0080909),
    1.0 :(0.0360,0.0080909)
}
#Rates.xlsx (foreign_rate, dmestic_rate)
usd_aud_rates = {
    1/12:(0.0419,0.03586666667),
    3/12:(0.0402,0.0356),
    6/12:(0.0381,0.0356),
    1.0 :(0.0360,0.0356)
}


# ============================================================
# Eq. (6): deterministic volatility backbone v_{0,t}
# ============================================================

def v0(t, V0, kappa, theta):
    return theta + (V0 - theta) * np.exp(-kappa * t)


# ============================================================
# Eq. (15): single-layer omega operator
# ============================================================

def omega_1(n, l_func, kappa, t, T):
    integrand = lambda u: np.exp(-n * kappa * (u - t)) * l_func(u)
    return np.exp(n * kappa * t) * quad(integrand, t, T)[0]


# ============================================================
# Eq. (16): two-layer omega operator
# ============================================================

def omega_2(n1, l1, n2, l2, kappa, T):
    integrand = lambda s: (
        np.exp(n1 * kappa * s)
        * l1(s)
        * omega_1(n2, l2, kappa, s, T)
    )
    return quad(integrand, 0.0, T)[0]


# ============================================================
# Eq. (17): three-layer omega operator
# ============================================================

def omega_3(n1, l1, n2, l2, n3, l3, kappa, T):
    integrand = lambda s: (
        np.exp(n1 * kappa * s)
        * l1(s)
        * omega_2(n2, l2, n3, l3, kappa, T - s)
    )
    return quad(integrand, 0.0, T)[0]


# ============================================================
# Eqs. (7)–(9): coefficient construction
# ============================================================

def compute_coefficients(V0, kappa, theta, lam, rho, T):

    psi = omega_1(
        0,
        lambda t: v0(t, V0, kappa, theta)**2,
        kappa, 0.0, T
    )

    a0 = omega_2(
        2,
        lambda t: lam**2 * v0(t, V0, kappa, theta)**2,
        -2,
        lambda t: 1.0,
        kappa, T
    )

    a1 = 2.0 * omega_2(
        1,
        lambda t: rho * lam * v0(t, V0, kappa, theta)**2,
        -1,
        lambda t: v0(t, V0, kappa, theta),
        kappa, T
    )

    a2 = 2.0 * omega_3(
        1,
        lambda t: rho * lam * v0(t, V0, kappa, theta)**2,
        1,
        lambda t: rho * lam * v0(t, V0, kappa, theta)**2,
        -2,
        lambda t: 1.0,
        kappa, T
    )

    b0 = 4.0 * omega_3(
        2,
        lambda t: lam**2 * v0(t, V0, kappa, theta)**2,
        -1,
        lambda t: v0(t, V0, kappa, theta),
        -1,
        lambda t: v0(t, V0, kappa, theta),
        kappa, T
    )

    b2 = 0.5 * a1**2

    return psi, a0, a1, a2, b0, b2


# ============================================================
# Eq. (10): Black–Scholes put (variance input)
# ============================================================

def BS_put(x, y, K, r, T):

    S0 = np.exp(x)
    vol = sqrt(y / T)

    d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ============================================================
# Eq. (5): final price expansion
# ============================================================

def price_expansion(x, psi, K, r, T, coeffs):

    a0, a1, a2, b0, b2 = coeffs
    # h = 1e-4
    h = 0.5

    P0 = BS_put(x, psi, K, r, T)

    d1 = (BS_put(x+h,psi,K,r,T)-BS_put(x-h,psi,K,r,T))/(2*h)
    d2 = (BS_put(x+h,psi,K,r,T)-2*P0+BS_put(x-h,psi,K,r,T))/h**2
    d3 = (BS_put(x+2*h,psi,K,r,T)-2*BS_put(x+h,psi,K,r,T)
          +2*BS_put(x-h,psi,K,r,T)-BS_put(x-2*h,psi,K,r,T))/(2*h**3)
    d4 = (BS_put(x+2*h,psi,K,r,T)-4*BS_put(x+h,psi,K,r,T)
          +6*P0-4*BS_put(x-h,psi,K,r,T)+BS_put(x-2*h,psi,K,r,T))/h**4

    return P0 + a0*d1 + a1*d2 + a2*d3 + b0*d2 + b2*d4


# ============================================================
# END OF PART (A)
# ============================================================


# ============================================================
# PART (B): IMPLIED VOLATILITY
# BLACK–SCHOLES OPTION PRICING FUNCTION
# ------------------------------------------------------------
# This function evaluates the Black–Scholes closed-form
# pricing formula for a European put option when the
# volatility parameter σ is given.
#
# Black, F. & Scholes, M. (1973)
#
# In the Black–Scholes framework the price of a European put
# option is given by:
#
#   P = K e^{-rT} Φ(-d2) − S0 Φ(-d1)
#
# where
#
#   d1 = [ ln(S0/K) + (r + σ²/2)T ] / (σ√T)
#   d2 = d1 − σ√T
#
# Φ(·) denotes the cumulative distribution function of the
# standard normal distribution.
#
# PURPOSE IN THIS MODEL
# The stochastic volatility expansion implemented in Part A
# produces theoretical option prices. In order to compare
# these prices to market conventions, we must express them as
# Black–Scholes implied volatilities. This function therefore
# computes the Black–Scholes price corresponding to a given
# volatility σ.
# ============================================================

def BS_put_from_vol(S0, K, r, T, sigma):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ============================================================
# IMPLIED VOLATILITY SOLVER
# IMPLIED VOLATILITY INVERSION
# This function computes the implied volatility corresponding
# to an observed option price by numerically inverting the
# Black–Scholes pricing formula.
#
# MATHEMATICAL BACKGROUND
# In financial markets, options are typically quoted in terms
# of implied volatility rather than price. Implied volatility
# is defined as the value of σ that satisfies:
#
#     P_market = BS_put(S0, K, r, T, σ)
#
# Because the Black–Scholes formula cannot be algebraically
# inverted for σ, a numerical root-finding method must be used.
#
# NUMERICAL METHOD
# Brent's method is used to solve the nonlinear equation:
#
#     f(σ) = BS_put(S0, K, r, T, σ) − P_model = 0
#
# Brent’s algorithm combines three techniques:
#
#   • Bisection method
#   • Secant method
#   • Inverse quadratic interpolation
#
# This makes it both stable and fast. 
#
# PURPOSE IN THIS MODEL
# The stochastic volatility expansion in Part A produces
# option prices. These prices are then converted into
# implied volatilities so that the results can be represented
# as an implied volatility surface σ(K,T), which is the
# standard representation used in options markets.
# ============================================================

def implied_vol_put(price, S0, K, r, T):

    if price <= 0:
        return np.nan

    def f(sigma):
        return BS_put_from_vol(S0, K, r, T, sigma) - price

    sigma_low = 1e-6
    sigma_high = 1.0

    f_low = f(sigma_low)
    f_high = f(sigma_high)

    while f_low * f_high > 0 and sigma_high < 10:
        sigma_high *= 2
        f_high = f(sigma_high)

    if f_low * f_high > 0:
        return np.nan

    return brentq(f, sigma_low, sigma_high)


# ============================================================
# BUILD IMPLIED VOL SURFACE
# This function constructs the implied volatility surface by
# evaluating implied volatilities across a grid of strikes and
# maturities.
#
# MATHEMATICAL CONCEPT
# The implied volatility surface is a function:
#
#     σ = σ(K, T)
#
# where
#
#   K = option strike price
#   T = option maturity
#
# For each point on this grid the following steps are performed:
#
#   1. Compute the theoretical option price using the
#      stochastic volatility perturbation expansion from
#      Part A.
#
#   2. Numerically invert the Black–Scholes pricing formula
#      to obtain the implied volatility corresponding to
#      that price.
#
#   3. Store the implied volatility value in the surface grid.
#
# PURPOSE IN THIS MODEL
# The resulting matrix of implied volatilities represents
# the full implied volatility surface σ(K,T), which captures
# how volatility varies with both strike (the volatility
# smile) and maturity (the volatility term structure).
# ============================================================

def build_surface(fx):

    S0 = fx["S0"]
    x0 = np.log(S0)
    V0 = fx["V0"]

    Ks = np.linspace(0.7*S0,1.3*S0,40)
    Ts = np.linspace(0.05,1.0,15)

    K_grid, T_grid = np.meshgrid(Ks, Ts)
    vol_grid = np.zeros_like(K_grid)

    for i in range(T_grid.shape[0]):
        for j in range(K_grid.shape[1]):

            T = T_grid[i,j]
            K = K_grid[i,j]

            T_key = min(fx["params"], key=lambda x: abs(x - T))

            kappa, theta, lam, rho = fx["params"][T_key]
            rf, rd = fx["rates"][T_key]

            r = rd - rf

            psi,a0,a1,a2,b0,b2 = compute_coefficients(
                V0,kappa,theta,lam,rho,T
            )

            price = price_expansion(
                x0,psi,K,r,T,(a0,a1,a2,b0,b2)
            )

            vol_grid[i,j] = implied_vol_put(price,S0,K,r,T)

    return K_grid, T_grid, vol_grid


# ============================================================
# SAVE CLEAN CSV DATA
# NUMERICAL EXPORT OF THE VOLATILITY SURFACE
# This function saves the numerical values of the implied
# volatility surface to a CSV file.
#
# Each row in the output file represents a point:
#
#     σ(K, T)
#
# where
#
#   Pair               FX currency pair
#   Maturity_Years     option maturity T
#   Strike             option strike K
#   Implied_Volatility σ(K,T)
# ============================================================

# Relative Volatiltiy Surface
def save_surface_data(K, T, vol, name):

    rows = []

    for i in range(T.shape[0]):
        for j in range(K.shape[1]):

            rows.append({
                "Pair": name,
                "Maturity_Years": T[i,j],
                "Strike": K[i,j],
                "Implied_Volatility": vol[i,j]
            })

    df = pd.DataFrame(rows)

    output_dir = os.path.expanduser("~/Desktop")
    filename = os.path.join(output_dir, f"{name}_surface_values.csv")

    df.to_csv(filename, index=False)

    print("Saved numerical surface:", filename)

# Absolute Volatility Surface
def save_absolute_surface_data(K, T, vol, name):

    rows = []

    for i in range(T.shape[0]):
        for j in range(K.shape[1]):

            rows.append({
                "Pair": name,
                "Maturity_Years": T[i,j],
                "Strike_Price": K[i,j],
                "Implied_Volatility": vol[i,j]
            })

    df = pd.DataFrame(rows)

    output_dir = os.path.expanduser("~/Desktop")
    filename = os.path.join(output_dir, f"{name}_absolute_surface_values.csv")

    df.to_csv(filename, index=False)

    print("Saved numerical absolute surface:", filename)


# ============================================================
# PLOT SURFACE
# IMPLIED VOLATILITY SURFACE VISUALIZATION
# This function visualizes the implied volatility surface
# σ(K,T) using a three-dimensional plot.
#
# In the plot:
#
#   x-axis : log-moneyness  log(K / S0)
#   y-axis : maturity T
#   z-axis : implied volatility σ
#
# Log-moneyness is commonly used instead of strike directly
# because it normalizes the strike relative to the underlying
# price and produces smoother volatility surfaces.
# ============================================================
# Relative Volatility Surface
def plot_surface(K, T, vol, name, S0):

    if np.all(np.isnan(vol)):
        vol[:] = 0.2
    else:
        vol = np.nan_to_num(vol, nan=np.nanmean(vol))

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    
    # Correct log-moneyness
    log_moneyness = np.log(K / S0)

    ax.plot_surface(log_moneyness, T, vol, edgecolor="none")

    ax.set_xlabel("Log-Moneyness ln(K/S0)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(name + " Log-Moneyness Surface")

    output_dir = os.path.expanduser("~/Desktop")
    filename = os.path.join(output_dir, f"{name}_log_moneyness_surface.png")

    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved figure:", filename)

# Absolute Volatility Surface

def plot_surface_absolute(K, T, vol, name):

    if np.all(np.isnan(vol)):
        vol[:] = 0.2
    else:
        vol = np.nan_to_num(vol, nan=np.nanmean(vol))

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    # Absolute strike surface
    ax.plot_surface(K, T, vol, edgecolor="none")

    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Maturity (T)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(name + " Absolute Volatility Surface")

    output_dir = os.path.expanduser("~/Desktop")
    filename = os.path.join(output_dir, f"{name}_absolute_vol_surface.png")

    plt.savefig(filename, dpi=150)
    plt.close()

    print("Saved figure:", filename)


# ============================================================
# FX DATA STRUCTURE, Sections 4.2.1, 4.2.2, 4.2.3
# ============================================================

fx_data = {

"USDAUD":{
"S0":spot_rates["USDAUD"],
"V0":0.0649,
"params":{
1/12:(4.19,0.0639,1.71,-0.40),
3/12:(2.33,0.1101,1.12,-0.74),
6/12:(2.26,0.1185,1.25,-0.73),
1.0:(1.80,0.1252,0.87,-0.92)
},
"rates":usd_aud_rates
},

"USDJPY":{
"S0":spot_rates["USDJPY"],
"V0":0.0442,
"params":{
1/12:(8.23,0.0796,2.47,-0.10),
3/12:(5.00,0.0647,1.32,-0.19),
6/12:(3.62,0.0932,1.61,-0.15),
1.0:(2.10,0.0674,1.88,-0.22)
},
"rates":usd_jpy_rates
},

#"USDSGD":{
#"S0":spot_rates["USDSGD"],
#"V0":0.0316,
#"params":{
#1/12:(2.90,0.0403,2.30,0.49),
#3/12:(2.85,0.0444,2.37,0.58),
#6/12:(2.76,0.0408,1.68,0.51),
#1.0:(2.81,0.0427,2.31,0.67)
#},
#"rates":usd_sgd_rates
#}
}


# ============================================================
# RUN ALL THREE SURFACES
# ============================================================

if __name__ == "__main__":

    for fx_name, fx in fx_data.items():

        print("Building implied volatility surface for", fx_name)

        K, T, vol = build_surface(fx)

        # Log-moneyness surface
        plot_surface(K, T, vol, fx_name, fx["S0"])

        # Absolute strike surface
        plot_surface_absolute(K, T, vol, fx_name)

        save_surface_data(K, T, vol, fx_name)
        save_absolute_surface_data(K, T, vol, fx_name)
        

    print("Done.")
