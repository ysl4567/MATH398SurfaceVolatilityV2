import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

from SurfaceVolatility import save_surface_data


def test_save_surface_data_writes_to_output_dir(tmp_path):
    import numpy as np

    K = np.array([[100.0]])
    T = np.array([[0.5]])
    vol = np.array([[0.2]])
    out = tmp_path / "output"
    save_surface_data(K, T, vol, "TEST", output_dir=str(out))
    assert (out / "TEST_surface_values.csv").exists()
