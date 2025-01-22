//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

// <mdspan>
//
// template<class OtherIndexType, size_t... OtherExtents>
//   friend constexpr bool operator==(const extents& lhs,
//                                    const extents<OtherIndexType, OtherExtents...>& rhs) noexcept;
//
// Returns: true if lhs.rank() equals rhs.rank() and
// if lhs.extent(r) equals rhs.extent(r) for every rank index r of rhs, otherwise false.
//

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class To, class From>
__host__ __device__ constexpr void test_comparison(bool equal, To dest, From src)
{
  ASSERT_NOEXCEPT(dest == src);
  assert((dest == src) == equal);
  assert((dest != src) == !equal);
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_different_rank()
{
  constexpr size_t D = cuda::std::dynamic_extent;

  test_comparison(false, cuda::std::extents<T1>(), cuda::std::extents<T2, D>(1));
  test_comparison(false, cuda::std::extents<T1>(), cuda::std::extents<T2, 1>());

  test_comparison(false, cuda::std::extents<T1, D>(1), cuda::std::extents<T2>());
  test_comparison(false, cuda::std::extents<T1, 1>(), cuda::std::extents<T2>());

  test_comparison(false, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D, D>(5, 5));
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5, D>(5));
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5, 1>());

  test_comparison(false, cuda::std::extents<T1, D, D>(5, 5), cuda::std::extents<T2, D>(5));
  test_comparison(false, cuda::std::extents<T1, 5, D>(5), cuda::std::extents<T2, D>(5));
  test_comparison(false, cuda::std::extents<T1, 5, 5>(), cuda::std::extents<T2, 5>());
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison_same_rank()
{
  constexpr size_t D = cuda::std::dynamic_extent;

  test_comparison(true, cuda::std::extents<T1>(), cuda::std::extents<T2>());

  test_comparison(true, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D>(5));
  test_comparison(true, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, D>(5));
  test_comparison(true, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, 5>());
  test_comparison(true, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 5>());
  test_comparison(false, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, D>(7));
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, D>(7));
  test_comparison(false, cuda::std::extents<T1, D>(5), cuda::std::extents<T2, 7>());
  test_comparison(false, cuda::std::extents<T1, 5>(), cuda::std::extents<T2, 7>());

  test_comparison(
    true, cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), cuda::std::extents<T2, D, D, D, D, D>(5, 6, 7, 8, 9));
  test_comparison(true, cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9), cuda::std::extents<T2, 5, D, D, 8, 9>(6, 7));
  test_comparison(true, cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), cuda::std::extents<T2, 5, 6, 7, 8, 9>());
  test_comparison(
    false, cuda::std::extents<T1, D, D, D, D, D>(5, 6, 7, 8, 9), cuda::std::extents<T2, D, D, D, D, D>(5, 6, 3, 8, 9));
  test_comparison(false, cuda::std::extents<T1, D, 6, D, 8, D>(5, 7, 9), cuda::std::extents<T2, 5, D, D, 3, 9>(6, 7));
  test_comparison(false, cuda::std::extents<T1, 5, 6, 7, 8, 9>(5, 6, 7, 8, 9), cuda::std::extents<T2, 5, 6, 7, 3, 9>());
}

template <class T1, class T2>
__host__ __device__ constexpr void test_comparison()
{
  test_comparison_same_rank<T1, T2>();
  test_comparison_different_rank<T1, T2>();
}

__host__ __device__ constexpr bool test()
{
  test_comparison<int, int>();
  test_comparison<int, size_t>();
  test_comparison<size_t, int>();
  test_comparison<size_t, long>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
