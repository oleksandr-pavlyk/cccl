//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib> // aligned_alloc
#include <cstring> // for memcpy
#include <iterator>
#include <memory> // for make_unique, unique_ptr

#include <cccl/c/types.h>
#include <stddef.h> // size_t
#include <stdint.h> // uint64_t

struct indirect_arg_t
{
  void* ptr;

  indirect_arg_t(cccl_iterator_t& it)
      : ptr(it.type == cccl_iterator_kind_t::CCCL_POINTER ? &it.state : it.state)
  {}

  indirect_arg_t(cccl_op_t& op)
      : ptr(op.type == cccl_op_kind_t::CCCL_STATELESS ? this : op.state)
  {}

  indirect_arg_t(cccl_value_t& val)
      : ptr(val.state)
  {}

  void* operator&() const
  {
    return ptr;
  }
};

class FreeDeleter
{
public:
  FreeDeleter()  = default;
  ~FreeDeleter() = default;

  template <typename U>
  void operator()(U* ptr) const
  {
    std::free(ptr);
  }
};

template <typename OffsetT>
struct indirect_host_incrementable_iterator_t
{
  static_assert(std::is_integral_v<OffsetT> && sizeof(OffsetT) == sizeof(uint64_t));

  using iterator_category = ::std::random_access_iterator_tag;
  using value_type        = void;
  using difference_type   = OffsetT;
  using pointer           = void*;
  using reference         = void*;

  void* state_ptr;
  OffsetT* host_offset_ptr;
  std::unique_ptr<void, FreeDeleter> owner;
  size_t value_size;
  size_t allocation_nbytes;

  indirect_host_incrementable_iterator_t(cccl_iterator_t& it)
      : state_ptr{}
      , host_offset_ptr{}
      , owner{}
      , value_size{}
      , allocation_nbytes{}
  {
    if (it.type == cccl_iterator_kind_t::CCCL_ITERATOR)
    {
      // we allocate memory to hold state and host_offset value of type uint64_t
      // the content of this allocation is to be copied by CUDA driver to the device
      const size_t offset_ptr_offset = align_up(it.size, sizeof(OffsetT));
      allocation_nbytes              = offset_ptr_offset + sizeof(OffsetT);

      owner = std::unique_ptr<void, FreeDeleter>(std::aligned_alloc(it.alignment, allocation_nbytes), FreeDeleter{});
      state_ptr = owner.get();
      // initialized host_offset variable to zero
      std::memset(state_ptr, 0, allocation_nbytes);

      host_offset_ptr = reinterpret_cast<OffsetT*>(reinterpret_cast<char*>(state_ptr) + offset_ptr_offset);
      std::memcpy(state_ptr, it.state, it.size);
    }
    else
    {
      state_ptr  = &it.state;
      value_size = it.value_type.size;
    }
  }

  indirect_host_incrementable_iterator_t(const indirect_host_incrementable_iterator_t& other)
      : state_ptr{}
      , host_offset_ptr{}
      , owner{}
      , value_size{}
      , allocation_nbytes{}
  {
    if (other.owner)
    {
      size_t alignment  = reinterpret_cast<uintptr_t>(other.state_ptr) & 63;
      allocation_nbytes = other.allocation_nbytes;
      owner     = std::unique_ptr<void, FreeDeleter>(std::aligned_alloc(alignment, allocation_nbytes), FreeDeleter{});
      state_ptr = owner.get();
      size_t relative_offset =
        (reinterpret_cast<char*>(other.host_offset_ptr) - reinterpret_cast<char*>(other.state_ptr));
      host_offset_ptr = reinterpret_cast<OffsetT*>(reinterpret_cast<char*>(state_ptr) + relative_offset);
      std::memcpy(state_ptr, other.owner.get(), allocation_nbytes);
    }
    else
    {
      state_ptr  = other.state_ptr;
      value_size = other.value_size;
    }
  }

  template <typename DiffT, std::enable_if_t<std::is_integral_v<DiffT> && sizeof(DiffT) == sizeof(OffsetT), bool> = true>
  indirect_host_incrementable_iterator_t& operator+=(DiffT offset)
  {
    if (host_offset_ptr)
    {
      // iterator kind: CCCL_ITERATOR
      DiffT* p = reinterpret_cast<DiffT*>(host_offset_ptr);
      *p += offset;
      return *this;
    }
    else
    {
      // iterator kind: CCCL_POINTER
      char** c_ptr = reinterpret_cast<char**>(state_ptr);
      *c_ptr += (offset * value_size);
      return *this;
    }
  }

  template <typename DiffT, std::enable_if_t<std::is_integral_v<DiffT> && sizeof(DiffT) == sizeof(OffsetT), bool> = true>
  indirect_host_incrementable_iterator_t operator+(DiffT offset) const
  {
    indirect_host_incrementable_iterator_t temp = *this;
    return temp += offset;
  }

  void* operator&() const
  {
    return state_ptr;
  }

  OffsetT get_offset() const
  {
    return *host_offset_ptr;
  }

private:
  template <typename IndexT>
  IndexT align_up(IndexT n, IndexT m)
  {
    return ((n + m - 1) / m) * m;
  }
};
