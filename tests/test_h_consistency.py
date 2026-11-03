import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

from SurfaceVolatility import H


def test_price_expansion_uses_h_0_01():
    assert H == 0.01
