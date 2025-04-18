//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day_last;

//  constexpr year_month_day_last(const chrono::year& y,
//                                const chrono::month_day_last& mdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_day_last by initializing
//                initializing y_ with y and mdl_ with mdl.
//
//  constexpr chrono::year                     year() const noexcept;
//  constexpr chrono::month                   month() const noexcept;
//  constexpr chrono::month_day_last month_day_last() const noexcept;
//  constexpr bool                               ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;

  static_assert(noexcept(year_month_day_last{year{1}, month_day_last{month{1}}}));

  constexpr month January = cuda::std::chrono::January;

  constexpr year_month_day_last ymdl0{year{}, month_day_last{month{}}};
  static_assert(ymdl0.year() == year{}, "");
  static_assert(ymdl0.month() == month{}, "");
  static_assert(ymdl0.month_day_last() == month_day_last{month{}}, "");
  static_assert(!ymdl0.ok(), "");

  constexpr year_month_day_last ymdl1{year{2019}, month_day_last{January}};
  static_assert(ymdl1.year() == year{2019}, "");
  static_assert(ymdl1.month() == January, "");
  static_assert(ymdl1.month_day_last() == month_day_last{January}, "");
  static_assert(ymdl1.ok(), "");

  return 0;
}
