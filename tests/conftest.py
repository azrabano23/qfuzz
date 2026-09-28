import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(autouse=True)
def _quiet_numpy():
    with np.errstate(all="ignore"):
        yield
