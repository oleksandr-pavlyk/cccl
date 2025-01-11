//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numbers>

// UNSUPPORTED: c++11

#include <cuda/std/numbers>

#include <test_macros.h>

template <class ExpectedT, class T>
__host__ __device__ constexpr bool test_defined(const T& value)
{
  ASSERT_SAME_TYPE(ExpectedT, T);

  const ExpectedT* addr = &value;
  unused(addr);

  return true;
}

__host__ __device__ constexpr bool test()
{
  test_defined<double>(cuda::std::numbers::e);
  test_defined<double>(cuda::std::numbers::log2e);
  test_defined<double>(cuda::std::numbers::log10e);
  test_defined<double>(cuda::std::numbers::pi);
  test_defined<double>(cuda::std::numbers::inv_pi);
  test_defined<double>(cuda::std::numbers::inv_sqrtpi);
  test_defined<double>(cuda::std::numbers::ln2);
  test_defined<double>(cuda::std::numbers::ln10);
  test_defined<double>(cuda::std::numbers::sqrt2);
  test_defined<double>(cuda::std::numbers::sqrt3);
  test_defined<double>(cuda::std::numbers::inv_sqrt3);
  test_defined<double>(cuda::std::numbers::egamma);
  test_defined<double>(cuda::std::numbers::phi);

  test_defined<float>(cuda::std::numbers::e_v<float>);
  test_defined<float>(cuda::std::numbers::log2e_v<float>);
  test_defined<float>(cuda::std::numbers::log10e_v<float>);
  test_defined<float>(cuda::std::numbers::pi_v<float>);
  test_defined<float>(cuda::std::numbers::inv_pi_v<float>);
  test_defined<float>(cuda::std::numbers::inv_sqrtpi_v<float>);
  test_defined<float>(cuda::std::numbers::ln2_v<float>);
  test_defined<float>(cuda::std::numbers::ln10_v<float>);
  test_defined<float>(cuda::std::numbers::sqrt2_v<float>);
  test_defined<float>(cuda::std::numbers::sqrt3_v<float>);
  test_defined<float>(cuda::std::numbers::inv_sqrt3_v<float>);
  test_defined<float>(cuda::std::numbers::egamma_v<float>);
  test_defined<float>(cuda::std::numbers::phi_v<float>);

  test_defined<double>(cuda::std::numbers::e_v<double>);
  test_defined<double>(cuda::std::numbers::log2e_v<double>);
  test_defined<double>(cuda::std::numbers::log10e_v<double>);
  test_defined<double>(cuda::std::numbers::pi_v<double>);
  test_defined<double>(cuda::std::numbers::inv_pi_v<double>);
  test_defined<double>(cuda::std::numbers::inv_sqrtpi_v<double>);
  test_defined<double>(cuda::std::numbers::ln2_v<double>);
  test_defined<double>(cuda::std::numbers::ln10_v<double>);
  test_defined<double>(cuda::std::numbers::sqrt2_v<double>);
  test_defined<double>(cuda::std::numbers::sqrt3_v<double>);
  test_defined<double>(cuda::std::numbers::inv_sqrt3_v<double>);
  test_defined<double>(cuda::std::numbers::egamma_v<double>);
  test_defined<double>(cuda::std::numbers::phi_v<double>);

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  test_defined<long double>(cuda::std::numbers::e_v<long double>);
  test_defined<long double>(cuda::std::numbers::log2e_v<long double>);
  test_defined<long double>(cuda::std::numbers::log10e_v<long double>);
  test_defined<long double>(cuda::std::numbers::pi_v<long double>);
  test_defined<long double>(cuda::std::numbers::inv_pi_v<long double>);
  test_defined<long double>(cuda::std::numbers::inv_sqrtpi_v<long double>);
  test_defined<long double>(cuda::std::numbers::ln2_v<long double>);
  test_defined<long double>(cuda::std::numbers::ln10_v<long double>);
  test_defined<long double>(cuda::std::numbers::sqrt2_v<long double>);
  test_defined<long double>(cuda::std::numbers::sqrt3_v<long double>);
  test_defined<long double>(cuda::std::numbers::inv_sqrt3_v<long double>);
  test_defined<long double>(cuda::std::numbers::egamma_v<long double>);
  test_defined<long double>(cuda::std::numbers::phi_v<long double>);
#endif // !defined(_LIBCUDACXX_NO_LONG_DOUBLE)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
