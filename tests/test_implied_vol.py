import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

import SurfaceVolatility as sv


def test_implied_vol_put_inverts_bs_put():
    price = sv.BS_put_from_vol(100.0, 100.0, 0.02, 1.0, 0.2)
    iv = sv.implied_vol_put(price, 100.0, 100.0, 0.02, 1.0)
    assert abs(iv - 0.2) < 1e-6
