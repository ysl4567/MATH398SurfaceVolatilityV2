import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

import SurfaceVolatility as sv


def test_price_expansion_round_trip():
    # ATM put with known params should be >= 0 and finite
    S0 = 100.0
    K = 100.0
    r = 0.02
    T = 1.0
    x = np.log(S0)
    V0 = 0.05
    kappa = 2.0
    theta = 0.05
    lam = 0.8
    rho = -0.7

    psi, a0, a1, a2, b0, b2 = sv.compute_coefficients(V0, kappa, theta, lam, rho, T)
    coeffs = (a0, a1, a2, b0, b2)

    price = sv.price_expansion(x, psi, K, r, T, coeffs)

    assert np.isfinite(price)
    assert price >= 0.0
