//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   constexpr void  // constexpr in C++20
//   push_heap(Iter first, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  T orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  for (int i = 1; i < 15; ++i)
  {
    cuda::std::push_heap(Iter(work), Iter(work + i), cuda::std::greater<T>());
    assert(cuda::std::is_permutation(work, work + i, orig));
    assert(cuda::std::is_heap(work, work + i, cuda::std::greater<T>()));
  }

  {
    T input[] = {5, 3, 4, 1, 2};
    cuda::std::push_heap(Iter(input), Iter(input + 1), cuda::std::greater<T>());
    assert(input[0] == 5);
    cuda::std::push_heap(Iter(input), Iter(input + 2), cuda::std::greater<T>());
    assert(input[0] == 3);
    cuda::std::push_heap(Iter(input), Iter(input + 3), cuda::std::greater<T>());
    assert(input[0] == 3);
    cuda::std::push_heap(Iter(input), Iter(input + 4), cuda::std::greater<T>());
    assert(input[0] == 1);
    cuda::std::push_heap(Iter(input), Iter(input + 5), cuda::std::greater<T>());
    assert(input[0] == 1);
    assert(cuda::std::is_heap(input, input + 5, cuda::std::greater<T>()));
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, random_access_iterator<int*>>();
  test<int, int*>();
  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
