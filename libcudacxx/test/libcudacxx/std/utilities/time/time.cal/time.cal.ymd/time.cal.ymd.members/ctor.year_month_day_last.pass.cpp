//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_day;

//  constexpr year_month_day(const year_month_day_last& ymdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_day by initializing
//              y_ with ymdl.year(), m_ with ymdl.month(), and d_ with ymdl.day().
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month               = cuda::std::chrono::month;
  using day                 = cuda::std::chrono::day;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using year_month_day      = cuda::std::chrono::year_month_day;

  static_assert(noexcept(year_month_day{cuda::std::declval<const year_month_day_last>()}));

  {
    constexpr year_month_day_last ymdl{year{2019}, month_day_last{month{1}}};
    constexpr year_month_day ymd{ymdl};

    static_assert(ymd.year() == year{2019}, "");
    static_assert(ymd.month() == month{1}, "");
    static_assert(ymd.day() == day{31}, "");
    static_assert(ymd.ok(), "");
  }

  {
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{month{4}}};
    constexpr year_month_day ymd{ymdl};

    static_assert(ymd.year() == year{1970}, "");
    static_assert(ymd.month() == month{4}, "");
    static_assert(ymd.day() == day{30}, "");
    static_assert(ymd.ok(), "");
  }

  {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{month{2}}};
    constexpr year_month_day ymd{ymdl};

    static_assert(ymd.year() == year{2000}, "");
    static_assert(ymd.month() == month{2}, "");
    static_assert(ymd.day() == day{29}, "");
    static_assert(ymd.ok(), "");
  }

  { // Feb 1900 was NOT a leap year.
    year_month_day_last ymdl{year{1900}, month_day_last{month{2}}};
    year_month_day ymd{ymdl};

    assert(ymd.year() == year{1900});
    assert(ymd.month() == month{2});
    assert(ymd.day() == day{28});
    assert(ymd.ok());
  }

  return 0;
}
