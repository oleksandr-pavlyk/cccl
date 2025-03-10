//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// A program that necessitates the instantiation of template optional for
// (possibly cv-qualified) in_place_t is ill-formed.

#include <cuda/std/optional>

int main(int, char**)
{
  using cuda::std::in_place;
  using cuda::std::in_place_t;
  using cuda::std::optional;

  optional<in_place_t> opt{}; // expected-note {{requested here}}
  // expected-error@optional:* {{instantiation of optional with in_place_t is ill-formed}}

  return 0;
}
