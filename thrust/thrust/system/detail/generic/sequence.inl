/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/sequence.h>
#include <thrust/tabulate.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ForwardIterator>
_CCCL_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last)
{
  using T = thrust::detail::it_value_t<ForwardIterator>;

  thrust::sequence(exec, first, last, T(0));
} // end sequence()

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, T init)
{
  thrust::sequence(exec, first, last, init, T(1));
} // end sequence()

namespace detail
{
template <typename T, typename = void>
struct compute_sequence_value
{
  T init;
  T step;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE T operator()(std::size_t i) const
  {
    return init + step * i;
  }
};
template <typename T>
struct compute_sequence_value<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
  T init;
  T step;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE T operator()(std::size_t i) const
  {
    return init + step * static_cast<T>(i);
  }
};
} // namespace detail

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE void
sequence(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, T init, T step)
{
  thrust::tabulate(
    exec, first, last, detail::compute_sequence_value<T>{::cuda::std::move(init), ::cuda::std::move(step)});
} // end sequence()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END
