"""Load the MNIST dataset from files."""

import gzip
import importlib
import importlib.resources
import io
import math

import numpy as np
from numpy.typing import NDArray

DTYPE_TABLE = {
    b"\x08": np.uint8,
    b"\x09": np.int8,
    b"\x0B": np.int16,
    b"\x0C": np.int32,
    b"\x0D": np.single,
    b"\x0E": np.double,
}


def idx_file_to_npy(idx_file: io.BufferedIOBase) -> NDArray:
    """Read the file format as described on LeCun's website and read it into a np array."""
    idx_file.seek(2)

    dtype_byte = idx_file.read(1)
    datatype = DTYPE_TABLE[dtype_byte]

    n_dims = int.from_bytes(idx_file.read(1), "big", signed=False)
    shape = tuple(
        [int.from_bytes(idx_file.read(4), "big", signed=False) for _ in range(n_dims)]
    )

    return np.frombuffer(
        idx_file.read(), count=math.prod(shape), dtype=datatype
    ).reshape(*shape)


def _load_mnist_file(filename: str) -> NDArray:
    filepath = importlib.resources.files(__spec__.parent).joinpath("data/" + filename)

    with gzip.open(str(filepath), "rb") as f:
        return idx_file_to_npy(f)


def mnist_train() -> tuple[NDArray, NDArray]:
    """Load the MNIST training set.

    Returns
        images, label: The training images and their corresponding label.

    """
    return _load_mnist_file("train-images-idx3-ubyte.gz"), _load_mnist_file(
        "train-labels-idx1-ubyte.gz"
    )


def mnist_test() -> tuple[NDArray, NDArray]:
    return _load_mnist_file("t10k-images-idx3-ubyte.gz"), _load_mnist_file(
        "t10k-labels-idx1-ubyte.gz"
    )
