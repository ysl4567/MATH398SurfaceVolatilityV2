import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

import SurfaceVolatility as sv


def test_compute_coefficients_finite():
    psi, a0, a1, a2, b0, b2 = sv.compute_coefficients(0.05, 2.0, 0.05, 0.8, -0.7, 1.0)
    assert all(np.isfinite(v) for v in (psi, a0, a1, a2, b0, b2))
