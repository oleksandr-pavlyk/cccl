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
#include <thrust/pair.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{

template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator unique(
  execution_policy<ExecutionPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred);

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
OutputIterator unique_copy(
  execution_policy<ExecutionPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
thrust::detail::it_difference_t<ForwardIterator> unique_count(
  execution_policy<DerivedPolicy>& exec, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred);

} // end namespace detail
} // end namespace tbb
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/unique.inl>
