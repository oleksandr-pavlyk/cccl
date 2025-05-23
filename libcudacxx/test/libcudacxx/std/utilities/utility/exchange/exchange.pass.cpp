//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// exchange

// template<class T, class U=T>
//    constexpr T            // constexpr after C++17
//    exchange(T& obj, U&& new_value)
//      noexcept(is_nothrow_move_constructible<T>::value && is_nothrow_assignable<T&, U>::value);

#include <cuda/std/cassert>
#include <cuda/std/utility>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  int v = 12;

  if (12 != cuda::std::exchange(v, 23) || v != 23)
  {
    return false;
  }

  if (23 != cuda::std::exchange(v, static_cast<short>(67)) || v != 67)
  {
    return false;
  }

  if (67 != cuda::std::exchange<int, short>(v, {}) || v != 0)
  {
    return false;
  }
  return true;
}

template <bool Move, bool Assign>
struct TestNoexcept
{
  TestNoexcept() = default;
  __host__ __device__ TestNoexcept(const TestNoexcept&);
  __host__ __device__ TestNoexcept(TestNoexcept&&) noexcept(Move);
  __host__ __device__ TestNoexcept& operator=(const TestNoexcept&);
  __host__ __device__ TestNoexcept& operator=(TestNoexcept&&) noexcept(Assign);
};

__host__ __device__ constexpr bool test_noexcept()
{
  {
    int x = 42;
    static_assert(noexcept(cuda::std::exchange(x, 42)));
    assert(x == 42);
  }
#if !TEST_COMPILER(NVHPC)
  {
    TestNoexcept<true, true> x{};
    static_assert(noexcept(cuda::std::exchange(x, cuda::std::move(x))));
    static_assert(!noexcept(cuda::std::exchange(x, x))); // copy-assignment is not noexcept
    unused(x);
  }
  {
    TestNoexcept<true, false> x{};
    static_assert(!noexcept(cuda::std::exchange(x, cuda::std::move(x))));
    unused(x);
  }
  {
    TestNoexcept<false, true> x{};
    static_assert(!noexcept(cuda::std::exchange(x, cuda::std::move(x))));
    unused(x);
  }
#endif // !TEST_COMPILER(NVHPC)

  return true;
}

int main(int, char**)
{
  {
    int v = 12;
    assert(cuda::std::exchange(v, 23) == 12);
    assert(v == 23);
    assert(cuda::std::exchange(v, static_cast<short>(67)) == 23);
    assert(v == 67);

    assert((cuda::std::exchange<int, short>(v, {})) == 67);
    assert(v == 0);
  }

  {
    bool b = false;
    assert(!cuda::std::exchange(b, true));
    assert(b);
  }

#ifdef _LIBCUDACXX_HAS_STRING
  {
    const cuda::std::string s1("Hi Mom!");
    const cuda::std::string s2("Yo Dad!");
    cuda::std::string s3 = s1; // Mom
    assert(cuda::std::exchange(s3, s2) == s1);
    assert(s3 == s2);
    assert(cuda::std::exchange(s3, "Hi Mom!") == s2);
    assert(s3 == s1);

    s3 = s2; // Dad
    assert(cuda::std::exchange(s3, {}) == s2);
    assert(s3.size() == 0);

    s3 = s2; // Dad
    assert(cuda::std::exchange(s3, "") == s2);
    assert(s3.size() == 0);
  }
#endif

  static_assert(test_constexpr(), "");

  static_assert(test_noexcept(), "");

  return 0;
}
