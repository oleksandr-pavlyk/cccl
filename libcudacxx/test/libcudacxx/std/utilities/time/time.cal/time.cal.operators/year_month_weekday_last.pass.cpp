//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday_last;

// constexpr year_month_weekday_last
//   operator/(const year_month& ym, const weekday_last& wdl) noexcept;
// Returns: {ym.year(), ym.month(), wdl}.
//
// constexpr year_month_weekday_last
//   operator/(const year& y, const month_weekday_last& mwdl) noexcept;
// Returns: {y, mwdl.month(), mwdl.weekday_last()}.
//
// constexpr year_month_weekday_last
//   operator/(int y, const month_weekday_last& mwdl) noexcept;
// Returns: year(y) / mwdl.
//
// constexpr year_month_weekday_last
//   operator/(const month_weekday_last& mwdl, const year& y) noexcept;
// Returns: y / mwdl.
//
// constexpr year_month_weekday_last
//   operator/(const month_weekday_last& mwdl, int y) noexcept;
// Returns: year(y) / mwdl.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year_month              = cuda::std::chrono::year_month;
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using month_weekday_last      = cuda::std::chrono::month_weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;

  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;
  constexpr month February  = cuda::std::chrono::February;

  { // operator/(const year_month& ym, const weekday_last& wdl) (and switched)
    constexpr year_month Feb2018{year{2018}, February};

    static_assert(noexcept(Feb2018 / weekday_last{Tuesday}));
    static_assert(cuda::std::is_same_v<year_month_weekday_last, decltype(Feb2018 / weekday_last{Tuesday})>);

    static_assert((Feb2018 / weekday_last{Tuesday}).year() == year{2018}, "");
    static_assert((Feb2018 / weekday_last{Tuesday}).month() == February, "");
    static_assert((Feb2018 / weekday_last{Tuesday}).weekday() == Tuesday, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 6; ++k)
        {
          year y{i};
          month m{j};
          weekday wd{k};
          year_month_weekday_last ymwdl = year_month{y, m} / weekday_last{wd};
          assert(ymwdl.year() == y);
          assert(ymwdl.month() == m);
          assert(ymwdl.weekday() == wd);
        }
      }
    }
  }

  { // operator/(const year& y, const month_weekday_last& mwdl) (and switched)
    constexpr month_weekday_last FebLastTues{February, weekday_last{Tuesday}};

    static_assert(noexcept(year{2018} / FebLastTues));
    static_assert(cuda::std::is_same_v<year_month_weekday_last, decltype(year{2018} / FebLastTues)>);
    static_assert(noexcept(FebLastTues / year{2018}));
    static_assert(cuda::std::is_same_v<year_month_weekday_last, decltype(FebLastTues / year{2018})>);

    static_assert((year{2018} / FebLastTues).year() == year{2018}, "");
    static_assert((year{2018} / FebLastTues).month() == February, "");
    static_assert((year{2018} / FebLastTues).weekday() == Tuesday, "");
    static_assert((FebLastTues / year{2018}).year() == year{2018}, "");
    static_assert((FebLastTues / year{2018}).month() == February, "");
    static_assert((FebLastTues / year{2018}).weekday() == Tuesday, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 6; ++k)
        {
          year y{i};
          month m{j};
          weekday wd{k};
          year_month_weekday_last ymwdl1 = y / month_weekday_last{m, weekday_last{wd}};
          year_month_weekday_last ymwdl2 = month_weekday_last{m, weekday_last{wd}} / y;
          assert(ymwdl1.year() == y);
          assert(ymwdl2.year() == y);
          assert(ymwdl1.month() == m);
          assert(ymwdl2.month() == m);
          assert(ymwdl1.weekday() == wd);
          assert(ymwdl2.weekday() == wd);
          assert(ymwdl1 == ymwdl2);
        }
      }
    }
  }

  { // operator/(int y, const month_weekday_last& mwdl) (and switched)
    constexpr month_weekday_last FebLastTues{February, weekday_last{Tuesday}};

    static_assert(noexcept(2018 / FebLastTues));
    static_assert(cuda::std::is_same_v<year_month_weekday_last, decltype(2018 / FebLastTues)>);
    static_assert(noexcept(FebLastTues / 2018));
    static_assert(cuda::std::is_same_v<year_month_weekday_last, decltype(FebLastTues / 2018)>);

    static_assert((2018 / FebLastTues).year() == year{2018}, "");
    static_assert((2018 / FebLastTues).month() == February, "");
    static_assert((2018 / FebLastTues).weekday() == Tuesday, "");
    static_assert((FebLastTues / 2018).year() == year{2018}, "");
    static_assert((FebLastTues / 2018).month() == February, "");
    static_assert((FebLastTues / 2018).weekday() == Tuesday, "");

    for (int i = 1000; i < 1010; ++i)
    {
      for (unsigned j = 1; j <= 12; ++j)
      {
        for (unsigned k = 0; k <= 6; ++k)
        {
          year y{i};
          month m{j};
          weekday wd{k};
          year_month_weekday_last ymwdl1 = i / month_weekday_last{m, weekday_last{wd}};
          year_month_weekday_last ymwdl2 = month_weekday_last{m, weekday_last{wd}} / i;
          assert(ymwdl1.year() == y);
          assert(ymwdl2.year() == y);
          assert(ymwdl1.month() == m);
          assert(ymwdl2.month() == m);
          assert(ymwdl1.weekday() == wd);
          assert(ymwdl2.weekday() == wd);
          assert(ymwdl1 == ymwdl2);
        }
      }
    }
  }

  return 0;
}
