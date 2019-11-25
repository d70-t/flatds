import os
import flatds
import numpy as np

BASEDIR = os.path.dirname(__file__)


def test_read():
    ds = flatds.open_flatds(os.path.join(BASEDIR, "test.flatds"))
    assert "test" in ds
    assert "bar" in ds
    assert np.all(ds.test.data == [1, 2, 3, 4, 5])
    assert np.all(ds.bar.data == [10, 20, 30, 40, 50])
