//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test also generates spurious warnings when instantiating std::span
// with a very large extent (like size_t(-2)) -- silence those.

// <span>

// template<size_t Offset, size_t Count = dynamic_extent>
//   constexpr span<element_type, see below> subspan() const;
//
//  Requires: offset <= size() &&
//            (count == dynamic_extent || count <= size() - offset)

#include <cuda/std/cstddef>
#include <cuda/std/span>

void f()
{
  int array[] = {1, 2, 3, 4};
  cuda::std::span<const int, 4> sp(array);

  //  Offset too large templatized
  [[maybe_unused]] auto s1 = sp.subspan<5>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset
                                              // out of range}}

  //  Count too large templatized
  [[maybe_unused]] auto s2 = sp.subspan<0, 5>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset
                                                 // + Count out of range}}

  //  Offset + Count too large templatized
  [[maybe_unused]] auto s3 = sp.subspan<2, 3>(); // expected-error@span:* {{span<T, N>::subspan<Offset, Count>(): Offset
                                                 // + Count out of range}}

  //  Offset + Count overflow templatized
  [[maybe_unused]] auto s4 = sp.subspan<3, cuda::std::size_t(-2)>(); // expected-error@span:* {{span<T,
                                                                     // N>::subspan<Offset, Count>(): Offset + Count out
                                                                     // of range}}, expected-error-re@span:* {{array is
                                                                     // too large{{(.* elements)}}}}
}
