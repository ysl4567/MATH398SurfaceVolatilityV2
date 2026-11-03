import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "OEX_project"))

from OEX_surface_comparison import load_npz


def test_load_npz_without_allow_pickle(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path,
             data_a=np.array([1.0]),
             data_b=np.array([2.0]),
             columns=["a", "b"],
             index=[0])
    df = load_npz(str(path))
    assert list(df.columns) == ["a", "b"]


def test_load_npz_pickled_object_array_raises_value_error(tmp_path):
    """Real OEX .npz files store `data` as a pickled object-dtype array.

    With ``allow_pickle`` off (hard constraint), accessing a pickled object
    column raises ``ValueError``. This documents the breaking change as
    expected behavior; such files must be re-saved with typed columns.
    """
    path = tmp_path / "pickled.npz"
    np.savez(path,
             data_a=np.array(["a", "b"], dtype=object),
             data_b=np.array([1.0, 2.0], dtype=object),
             columns=["a", "b"],
             index=[0, 1])
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_npz(str(path))
