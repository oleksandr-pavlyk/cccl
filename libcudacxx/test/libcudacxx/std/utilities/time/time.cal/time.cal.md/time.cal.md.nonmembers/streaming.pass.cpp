//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: *

// <chrono>
// class month_day;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day& md);
//
//     Returns: os << md.month() << '/' << md.day().
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const month_day& md);
//
// Effects: Streams md into os using the format specified by the NTCTS fmt.
//          fmt encoding follows the rules specified in 25.11.

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using month_day = cuda::std::chrono::month_day;
  using month     = cuda::std::chrono::month;
  using day       = cuda::std::chrono::day;
  std::cout << month_day{month{1}, day{1}};

  return 0;
}
