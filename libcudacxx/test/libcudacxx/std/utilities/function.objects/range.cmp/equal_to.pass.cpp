//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// ranges::equal_to

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "compare_types.h"
#include "MoveOnly.h"
#include "pointer_comparison_test_helper.h"
#include "test_macros.h"

struct NotEqualityComparable
{
  __host__ __device__ friend bool operator==(const NotEqualityComparable&, const NotEqualityComparable&);
  __host__ __device__ friend bool operator!=(const NotEqualityComparable&, const NotEqualityComparable&) = delete;
};

static_assert(!cuda::std::is_invocable_v<cuda::std::ranges::equal_to, NotEqualityComparable, NotEqualityComparable>);
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC considers implicit conversions in C++17
static_assert(!cuda::std::is_invocable_v<cuda::std::ranges::equal_to, int, MoveOnly>);
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(cuda::std::is_invocable_v<cuda::std::ranges::equal_to, explicit_operators, explicit_operators>);

#if TEST_STD_VER > 2017
static_assert(requires { typename cuda::std::ranges::equal_to::is_transparent; });
#else
template <class T, class = void>
inline constexpr bool is_transparent = false;
template <class T>
inline constexpr bool is_transparent<T, cuda::std::void_t<typename T::is_transparent>> = true;
static_assert(is_transparent<cuda::std::ranges::equal_to>);
#endif

__host__ __device__ constexpr bool test()
{
  auto fn = cuda::std::ranges::equal_to();

  assert(fn(MoveOnly(42), MoveOnly(42)));

  ForwardingTestObject a{};
  ForwardingTestObject b{};
  assert(!fn(a, b));
  assert(fn(cuda::std::move(a), cuda::std::move(b)));

  assert(!fn(1, 2));
  assert(!fn(2, 1));
  assert(fn(2, 2));

  assert(!fn(2, 1L));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  // test total ordering of int* for equal_to<int*> and equal_to<void>.
  do_pointer_comparison_test(cuda::std::ranges::equal_to());

  return 0;
}
