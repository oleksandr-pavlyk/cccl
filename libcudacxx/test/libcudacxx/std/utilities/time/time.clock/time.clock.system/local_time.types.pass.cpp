//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++17

// <cuda/std/chrono>

// struct local_t {};
// template<class Duration>
//   using local_time  = time_point<system_clock, Duration>;
// using local_seconds = sys_time<seconds>;
// using local_days    = sys_time<days>;

// [Example:
//   sys_seconds{sys_days{1970y/January/1}}.time_since_epoch() is 0s.
//   sys_seconds{sys_days{2000y/January/1}}.time_since_epoch() is 946’684’800s, which is 10’957 * 86’400s.
// —end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
#ifndef __cuda_std__
  using local_t = cuda::std::chrono::local_t;
  using year    = cuda::std::chrono::year;

  using seconds = cuda::std::chrono::seconds;
  using minutes = cuda::std::chrono::minutes;
  using days    = cuda::std::chrono::days;

  using local_seconds = cuda::std::chrono::local_seconds;
  using local_minutes = cuda::std::chrono::local_time<minutes>;
  using local_days    = cuda::std::chrono::local_days;

  constexpr cuda::std::chrono::month January = cuda::std::chrono::January;

  static_assert(cuda::std::is_same_v<cuda::std::chrono::local_time<seconds>, local_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::local_time<days>, local_days>);

  //  Test the long form, too
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<local_t, seconds>, local_seconds>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<local_t, minutes>, local_minutes>);
  static_assert(cuda::std::is_same_v<cuda::std::chrono::time_point<local_t, days>, local_days>);

  //  Test some well known values
  local_days d0 = local_days{year{1970} / January / 1};
  local_days d1 = local_days{year{2000} / January / 1};
  static_assert(cuda::std::is_same_v<decltype(d0.time_since_epoch()), days>);
  assert(d0.time_since_epoch().count() == 0);
  assert(d1.time_since_epoch().count() == 10957);

  local_seconds s0{d0};
  local_seconds s1{d1};
  static_assert(cuda::std::is_same_v<decltype(s0.time_since_epoch()), seconds>);
  assert(s0.time_since_epoch().count() == 0);
  assert(s1.time_since_epoch().count() == 946684800L);
#endif // __cuda_std__

  return 0;
}
