//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H

#include <cuda/std/utility>

#include "CommonHelpers.h"
#include "test_macros.h"

// Idiosyncratic element type for mdspan
// Make sure we don't assume copyable, default constructible, movable etc.
struct MinimalElementType
{
  int val;
  constexpr MinimalElementType()                                     = delete;
  constexpr MinimalElementType(const MinimalElementType&)            = delete;
  constexpr MinimalElementType& operator=(const MinimalElementType&) = delete;
  __host__ __device__ constexpr explicit MinimalElementType(int v) noexcept
      : val(v)
  {}
};

// Helper class to create pointer to MinimalElementType
template <class T, size_t N>
struct ElementPool
{
private:
  __host__ __device__ static constexpr int to_42(const int) noexcept
  {
    return 42;
  }

  template <int... Indices>
  __host__ __device__ constexpr ElementPool(cuda::std::integer_sequence<int, Indices...>)
      : ptr_{T(to_42(Indices))...}
  {}

public:
  __host__ __device__ constexpr ElementPool()
      : ElementPool(cuda::std::make_integer_sequence<int, N>())
  {}

  __host__ __device__ constexpr T* get_ptr()
  {
    return ptr_;
  }

private:
  T ptr_[N];
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
