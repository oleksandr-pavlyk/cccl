//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_polymorphic

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_polymorphic()
{
  static_assert(cuda::std::is_polymorphic<T>::value, "");
  static_assert(cuda::std::is_polymorphic<const T>::value, "");
  static_assert(cuda::std::is_polymorphic<volatile T>::value, "");
  static_assert(cuda::std::is_polymorphic<const volatile T>::value, "");
  static_assert(cuda::std::is_polymorphic_v<T>, "");
  static_assert(cuda::std::is_polymorphic_v<const T>, "");
  static_assert(cuda::std::is_polymorphic_v<volatile T>, "");
  static_assert(cuda::std::is_polymorphic_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_polymorphic()
{
  static_assert(!cuda::std::is_polymorphic<T>::value, "");
  static_assert(!cuda::std::is_polymorphic<const T>::value, "");
  static_assert(!cuda::std::is_polymorphic<volatile T>::value, "");
  static_assert(!cuda::std::is_polymorphic<const volatile T>::value, "");
  static_assert(!cuda::std::is_polymorphic_v<T>, "");
  static_assert(!cuda::std::is_polymorphic_v<const T>, "");
  static_assert(!cuda::std::is_polymorphic_v<volatile T>, "");
  static_assert(!cuda::std::is_polymorphic_v<const volatile T>, "");
}

class Empty
{};

class NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  __host__ __device__ virtual ~Abstract() = 0;
};

class Final final
{};

int main(int, char**)
{
  test_is_not_polymorphic<void>();
  test_is_not_polymorphic<int&>();
  test_is_not_polymorphic<int>();
  test_is_not_polymorphic<double>();
  test_is_not_polymorphic<int*>();
  test_is_not_polymorphic<const int*>();
  test_is_not_polymorphic<char[3]>();
  test_is_not_polymorphic<char[]>();
  test_is_not_polymorphic<Union>();
  test_is_not_polymorphic<Empty>();
  test_is_not_polymorphic<bit_zero>();
  test_is_not_polymorphic<Final>();
  test_is_not_polymorphic<NotEmpty&>();
  test_is_not_polymorphic<Abstract&>();

  test_is_polymorphic<NotEmpty>();
  test_is_polymorphic<Abstract>();

  return 0;
}
