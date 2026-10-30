import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

from OEX_surface_comparison import load_npz


def test_load_npz_without_allow_pickle(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, data=np.array([[1.0, 2.0]]), columns=["a", "b"], index=[0])
    df = load_npz(str(path))
    assert list(df.columns) == ["a", "b"]
