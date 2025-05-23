//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T, class Alloc> struct uses_allocator;

#include <cuda/std/__memory_>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR
#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

struct B
{
  typedef int allocator_type;
};

#if !TEST_COMPILER(NVRTC)
struct C
{
  static int allocator_type;
};
#endif // !TEST_COMPILER(NVRTC)

struct D
{
  __host__ __device__ static int allocator_type()
  {
    return 0;
  }
};

struct E
{
private:
  typedef int allocator_type;
};

template <bool Expected, class T, class A>
__host__ __device__ void test()
{
  static_assert((cuda::std::uses_allocator<T, A>::value == Expected), "");
  static_assert(
    cuda::std::is_base_of<cuda::std::integral_constant<bool, Expected>, cuda::std::uses_allocator<T, A>>::value, "");

  static_assert(cuda::std::is_same_v<decltype(cuda::std::uses_allocator_v<T, A>), const bool>);
  static_assert((cuda::std::uses_allocator_v<T, A> == Expected), "");
}

int main(int, char**)
{
  test<false, int, cuda::std::allocator<int>>();
#if defined(_LIBCUDACXX_HAS_VECTOR)
  test<true, cuda::std::vector<int>, cuda::std::allocator<int>>();
#endif //_LIBCUDACXX_HAS_VECTOR
  test<false, A, cuda::std::allocator<int>>();
  test<false, B, cuda::std::allocator<int>>();
  test<true, B, double>();
#if !TEST_COMPILER(NVRTC)
  test<false, C, decltype(C::allocator_type)>();
#endif // !TEST_COMPILER(NVRTC)
  test<false, D, decltype(D::allocator_type)>();
#if !TEST_COMPILER(GCC) // E::allocator_type is private
  test<false, E, int>();
#endif // !TEST_COMPILER(GCC)

  static_assert((!cuda::std::uses_allocator<int, cuda::std::allocator<int>>::value), "");
#if defined(_LIBCUDACXX_HAS_VECTOR)
  static_assert((cuda::std::uses_allocator<cuda::std::vector<int>, cuda::std::allocator<int>>::value), "");
#endif // _LIBCUDACXX_HAS_VECTOR
  static_assert((!cuda::std::uses_allocator<A, cuda::std::allocator<int>>::value), "");
  static_assert((!cuda::std::uses_allocator<B, cuda::std::allocator<int>>::value), "");
  static_assert((cuda::std::uses_allocator<B, double>::value), "");
#if !TEST_COMPILER(NVRTC)
  static_assert((!cuda::std::uses_allocator<C, decltype(C::allocator_type)>::value), "");
  static_assert((!cuda::std::uses_allocator<D, decltype(D::allocator_type)>::value), "");
#endif // !TEST_COMPILER(NVRTC)
#if !TEST_COMPILER(GCC) // E::allocator_type is private
  static_assert((!cuda::std::uses_allocator<E, int>::value), "");
#endif // !TEST_COMPILER(GCC)

  return 0;
}
