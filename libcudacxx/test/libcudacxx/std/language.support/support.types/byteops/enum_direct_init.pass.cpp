//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// The following compilers don't like "cuda::std::byte b1{1}"
// XFAIL: clang-3.5, clang-3.6, clang-3.7, clang-3.8
// XFAIL: apple-clang-6, apple-clang-7, apple-clang-8.0
// UNSUPPORTED: gcc-6

int main(int, char**)
{
  constexpr cuda::std::byte b{42};
  static_assert(cuda::std::to_integer<int>(b) == 42, "");

  return 0;
}
