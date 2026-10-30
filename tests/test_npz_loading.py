import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

from OEX_surface_comparison import load_npz


def test_load_npz_without_allow_pickle(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, data=np.array([[1.0, 2.0]]), columns=["a", "b"], index=[0])
    df = load_npz(str(path))
    assert list(df.columns) == ["a", "b"]


def test_load_npz_pickled_object_array_raises_value_error(tmp_path):
    """Real OEX .npz files store `data` as a pickled object-dtype array.

    With ``allow_pickle`` off (hard constraint), ``np.load`` raises
    ``ValueError`` when the object array is accessed. This documents the
    breaking change as expected behavior; such files must be re-saved with
    typed (non-object) columns before loading.
    """
    path = tmp_path / "pickled.npz"
    np.savez(path, data=np.array([["a", 1.0], ["b", 2.0]], dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_npz(str(path))
