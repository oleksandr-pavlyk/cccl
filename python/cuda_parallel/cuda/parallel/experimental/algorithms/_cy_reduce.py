# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations  # TODO: required for Python 3.7 docs env

from typing import Callable

import numba
import numpy as np
from numba.cuda.cudadrv import enums

from .. import _cccl_for_cy as cccl
from .. import _cy_bindings as cyb
from .._caching import CachableFunction, cache_with_key
from .._cccl_for_cy import call_build
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike, GpuStruct


class _Reduce:
    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
    ):
        # Referenced from __del__:
        self.build_result = None

        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        self.h_init_cccl = cccl.to_cccl_value(h_init)
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)
        self.build_result = cyb.DeviceReduceBuildResult()
        error = call_build(
            cyb.device_reduce_build,
            self.build_result,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
            self.h_init_cccl,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building reduce")

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        set_state_fn = cccl.set_cccl_iterator_state
        set_state_fn(self.d_in_cccl, d_in)
        set_state_fn(self.d_out_cccl, d_out)

        self.h_init_cccl.state = cccl.to_cccl_value_state(h_init)

        stream_handle = protocols.validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = None
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = protocols.get_data_pointer(temp_storage)

        error, temp_storage_bytes = cyb.device_reduce(
            self.build_result,
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_wrapper,
            self.h_init_cccl,
            stream_handle,
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes

    def __del__(self):
        if self.build_result is None:
            return
        cyb.device_reduce_cleanup(self.build_result)


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
    h_init: np.ndarray,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = protocols.get_dtype(d_out)
    op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (d_in_key, d_out_key, op_key, h_init_key)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``reduce_into`` is used to compute the minimum value of a sequence of integers.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-min
            :end-before: example-end reduce-min

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array (of size 1) that will store the result of the reduction
        op: Callable representing the binary operator to apply
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    return _Reduce(d_in, d_out, op, h_init)
