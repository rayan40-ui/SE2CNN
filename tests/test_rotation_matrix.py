from importlib.machinery import SourceFileLoader
from pathlib import Path
import numpy as np

rotation_matrix = SourceFileLoader(
    "rotation_matrix", str(Path(__file__).resolve().parent.parent / "se2cnn" / "rotation_matrix.py")
).load_module()
ToLinearIndex = rotation_matrix.ToLinearIndex
RotationOperatorMatrixSparse = rotation_matrix.RotationOperatorMatrixSparse


def test_to_linear_index_rectangular():
    assert ToLinearIndex([1, 2], [2, 3]) == 1 * 3 + 2


def test_rotation_operator_matrix_sparse_identity():
    idx, vals = RotationOperatorMatrixSparse([2, 3], 0.0, diskMask=False)
    size = 2 * 3
    dense = np.zeros((size, size))
    for (row, col), val in zip(idx, vals):
        dense[row, col] = val
    assert np.allclose(dense, np.eye(size))
