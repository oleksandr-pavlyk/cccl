//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// struct output_iterator_tag {};

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  [[maybe_unused]] cuda::std::output_iterator_tag tag;

  static_assert((!cuda::std::is_base_of<cuda::std::input_iterator_tag, cuda::std::output_iterator_tag>::value), "");

  return 0;
}
