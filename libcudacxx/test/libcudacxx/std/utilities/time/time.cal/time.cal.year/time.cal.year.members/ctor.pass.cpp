//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

//                     year() = default;
//  explicit constexpr year(int m) noexcept;
//  explicit constexpr operator int() const noexcept;

//  Effects: Constructs an object of type year by initializing y_ with y.
//    The value held is unspecified if d is not in the range [0, 255].

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(year{}));
  static_assert(noexcept(year(0U)));
  static_assert(noexcept(static_cast<int>(year(0U))));

  constexpr year y0{};
  static_assert(static_cast<int>(y0) == 0, "");

  constexpr year y1{1};
  static_assert(static_cast<int>(y1) == 1, "");

  for (int i = 0; i <= 2550; i += 7)
  {
    year year(i);
    assert(static_cast<int>(year) == i);
  }

  return 0;
}
