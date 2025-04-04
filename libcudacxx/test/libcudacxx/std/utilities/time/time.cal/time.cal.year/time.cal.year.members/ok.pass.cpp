//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr bool ok() const noexcept;
//  Returns: min() <= y_ && y_ <= max().
//
//  static constexpr year min() noexcept;
//   Returns year{ 32767};
//  static constexpr year max() noexcept;
//   Returns year{-32767};

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(cuda::std::declval<const year>().ok()));
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::declval<const year>().ok())>);

  static_assert(noexcept(year::max()));
  static_assert(cuda::std::is_same_v<year, decltype(year::max())>);

  static_assert(noexcept(year::min()));
  static_assert(cuda::std::is_same_v<year, decltype(year::min())>);

  static_assert(static_cast<int>(year::min()) == -32767, "");
  static_assert(static_cast<int>(year::max()) == 32767, "");

  assert(year{-20001}.ok());
  assert(year{-2000}.ok());
  assert(year{-1}.ok());
  assert(year{0}.ok());
  assert(year{1}.ok());
  assert(year{2000}.ok());
  assert(year{20001}.ok());

  static_assert(!year{-32768}.ok(), "");

  return 0;
}
