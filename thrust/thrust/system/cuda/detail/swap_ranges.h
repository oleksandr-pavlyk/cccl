/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/distance.h>
#  include <thrust/swap.h>
#  include <thrust/system/cuda/detail/par_to_seq.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/transform.h>

#  include <cuda/std/utility>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{

namespace __swap_ranges
{

template <class ItemsIt1, class ItemsIt2>
struct swap_f
{
  ItemsIt1 items1;
  ItemsIt2 items2;

  using value1_type = thrust::detail::it_value_t<ItemsIt1>;
  using value2_type = thrust::detail::it_value_t<ItemsIt2>;

  THRUST_FUNCTION
  swap_f(ItemsIt1 items1_, ItemsIt2 items2_)
      : items1(items1_)
      , items2(items2_)
  {}

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    // TODO(bgruber): this should probably use ::cuda::std::iter_swap(items1 + idx, items2 + idx);
    value1_type item1 = items1[idx];
    value2_type item2 = items2[idx];
    using ::cuda::std::swap;
    swap(item1, item2);
    items1[idx] = item1;
    items2[idx] = item2;
  }
};
} // namespace __swap_ranges

template <class Derived, class ItemsIt1, class ItemsIt2>
ItemsIt2 _CCCL_HOST_DEVICE
swap_ranges(execution_policy<Derived>& policy, ItemsIt1 first1, ItemsIt1 last1, ItemsIt2 first2)
{
  using size_type = thrust::detail::it_difference_t<ItemsIt1>;

  size_type num_items = static_cast<size_type>(::cuda::std::distance(first1, last1));

  cuda_cub::parallel_for(policy, __swap_ranges::swap_f<ItemsIt1, ItemsIt2>(first1, first2), num_items);

  return first2 + num_items;
}

} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
