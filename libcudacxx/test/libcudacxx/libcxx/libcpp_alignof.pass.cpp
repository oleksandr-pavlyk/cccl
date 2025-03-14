// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCUDACXX_ALIGNOF acts the same as the C++11 keyword `alignof`, and
// not as the GNU extension `__alignof`. The former returns the minimal required
// alignment for a type, whereas the latter returns the preferred alignment.
//
// See llvm.org/PR39713

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  static_assert(_LIBCUDACXX_ALIGNOF(T) == cuda::std::alignment_of<T>::value, "");
  static_assert(_LIBCUDACXX_ALIGNOF(T) == alignof(T), "");
  static_assert(_LIBCUDACXX_ALIGNOF(T) == alignof(T), "");
#if TEST_COMPILER(CLANG)
  static_assert(_LIBCUDACXX_ALIGNOF(T) == _Alignof(T), "");
#endif // TEST_COMPILER(CLANG)
}

int main(int, char**)
{
  test<int>();
  test<long long>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  return 0;
}
