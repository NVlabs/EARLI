# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Restricted deserialization helpers.

``pickle.load`` and ``torch.load(weights_only=False)`` execute arbitrary code
embedded in the file being loaded. The helpers here only reconstruct plain
data types (builtins, numpy arrays, torch tensors), so a tampered checkpoint
or data file cannot run code. Requires torch>=2.6.
"""

import pickle
import uuid

import numpy as np
import torch

try:  # numpy >= 2
    from numpy._core.multiarray import _reconstruct, scalar
except ImportError:  # numpy < 2
    from numpy.core.multiarray import _reconstruct, scalar

# Data-only classes that appear in EARLI checkpoints, beyond torch's defaults.
# The (obj, 'name') tuples map the pre-numpy-2 module path stored in old files.
_SAFE_TORCH_GLOBALS = [
    np.ndarray,
    np.dtype,
    uuid.UUID,
    (scalar, 'numpy.core.multiarray.scalar'),
    (_reconstruct, 'numpy.core.multiarray._reconstruct'),
]
try:
    import numpy.dtypes as _np_dtypes
    _SAFE_TORCH_GLOBALS += [getattr(_np_dtypes, _name) for _name in dir(_np_dtypes)
                            if _name.endswith('DType')]
except ImportError:  # numpy < 1.25
    pass


def safe_torch_load(filename, **kwargs):
    """torch.load restricted to tensors and plain data (weights_only=True)."""
    kwargs.setdefault('map_location', None)
    with torch.serialization.safe_globals(_SAFE_TORCH_GLOBALS):
        return torch.load(filename, weights_only=True, **kwargs)


# (module, qualname) pairs allowed in data pickle files. builtins are handled
# separately below. torch.storage._load_from_bytes delegates to torch.load,
# which is itself restricted (weights_only defaults to True in torch>=2.6).
_SAFE_PICKLE_GLOBALS = {
    ('collections', 'OrderedDict'),
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'),
    ('numpy._core.multiarray', '_reconstruct'),
    ('numpy._core.multiarray', 'scalar'),
    ('torch._utils', '_rebuild_tensor_v2'),
    ('torch.storage', '_load_from_bytes'),
}

_SAFE_BUILTINS = {
    'bool', 'bytearray', 'bytes', 'complex', 'dict', 'float', 'frozenset',
    'int', 'list', 'range', 'set', 'slice', 'str', 'tuple',
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'builtins' and name in _SAFE_BUILTINS:
            return super().find_class(module, name)
        if (module, name) in _SAFE_PICKLE_GLOBALS:
            return super().find_class(module, name)
        if module == 'numpy.dtypes' and name.endswith('DType'):
            return super().find_class(module, name)
        if module == 'torch' and (name == 'Tensor' or name.endswith('Storage')):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f'{module}.{name} is not allowed in data files; if it is legitimate, '
            f'add it to _SAFE_PICKLE_GLOBALS in {__name__}')


def safe_pickle_load(file_obj):
    """pickle.load restricted to plain data types (no code execution)."""
    return _RestrictedUnpickler(file_obj).load()
