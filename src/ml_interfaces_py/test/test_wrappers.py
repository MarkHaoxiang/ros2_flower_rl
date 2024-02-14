from typing import Sequence
import pytest

import numpy as np
import torch

from ml_interfaces_py.lib import FloatTensor

class TestFloatTensor:

    @pytest.parametrize("shape", [(1,), (2,3)])
    def test_construction(self, shape: Sequence[int]):
        # Numpy
        a: np.ndarray = np.random.rand(shape=shape)
        a_ = FloatTensor.from_numpy(a)
        assert np.array(a_.values) == a.flatten()

        # Torch
        b = torch.rand(size=shape)
        b_ = FloatTensor.from_torch(b)
        assert torch.tensor(b_.values()) == b.flatten()
