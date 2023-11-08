//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SEARCH_H
#define _LIBCUDACXX___ALGORITHM_SEARCH_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#include "../__algorithm/comp.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__iterator/advance.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/enable_if.h"
#include "../__utility/pair.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _BinaryPredicate, class _ForwardIterator1, class _ForwardIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_ForwardIterator1, _ForwardIterator1>
  __search(_ForwardIterator1 __first1,
           _ForwardIterator1 __last1,
           _ForwardIterator2 __first2,
           _ForwardIterator2 __last2,
           _BinaryPredicate __pred,
           forward_iterator_tag,
           forward_iterator_tag)
{
  if (__first2 == __last2)
  {
    return _CUDA_VSTD::make_pair(__first1, __first1); // Everything matches an empty sequence
  }
  while (true)
  {
    // Find first element in sequence 1 that matchs *__first2, with a mininum of loop checks
    while (true)
    {
      if (__first1 == __last1) // return __last1 if no element matches *__first2
      {
        return _CUDA_VSTD::make_pair(__last1, __last1);
      }
      if (__pred(*__first1, *__first2))
      {
        break;
      }
      ++__first1;
    }
    // *__first1 matches *__first2, now match elements after here
    _ForwardIterator1 __m1 = __first1;
    _ForwardIterator2 __m2 = __first2;
    while (true)
    {
      if (++__m2 == __last2) // If pattern exhausted, __first1 is the answer (works for 1 element pattern)
      {
        return _CUDA_VSTD::make_pair(__first1, __m1);
      }
      if (++__m1 == __last1) // Otherwise if source exhaused, pattern not found
      {
        return _CUDA_VSTD::make_pair(__last1, __last1);
      }
      if (!__pred(*__m1, *__m2)) // if there is a mismatch, restart with a new __first1
      {
        ++__first1;
        break;
      } // else there is a match, check next elements
    }
  }
}

template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_RandomAccessIterator1, _RandomAccessIterator1>
  __search(_RandomAccessIterator1 __first1,
           _RandomAccessIterator1 __last1,
           _RandomAccessIterator2 __first2,
           _RandomAccessIterator2 __last2,
           _BinaryPredicate __pred,
           random_access_iterator_tag,
           random_access_iterator_tag)
{
  typedef typename iterator_traits<_RandomAccessIterator1>::difference_type _Diff1;
  typedef typename iterator_traits<_RandomAccessIterator2>::difference_type _Diff2;
  // Take advantage of knowing source and pattern lengths.  Stop short when source is smaller than pattern
  const _Diff2 __len2 = __last2 - __first2;
  if (__len2 == 0)
  {
    return _CUDA_VSTD::make_pair(__first1, __first1);
  }
  const _Diff1 __len1 = __last1 - __first1;
  if (__len1 < __len2)
  {
    return _CUDA_VSTD::make_pair(__last1, __last1);
  }
  const _RandomAccessIterator1 __s = __last1 - (__len2 - 1); // Start of pattern match can't go beyond here

  while (true)
  {
    while (true)
    {
      if (__first1 == __s)
      {
        return _CUDA_VSTD::make_pair(__last1, __last1);
      }
      if (__pred(*__first1, *__first2))
      {
        break;
      }
      ++__first1;
    }

    _RandomAccessIterator1 __m1 = __first1;
    _RandomAccessIterator2 __m2 = __first2;
    while (true)
    {
      if (++__m2 == __last2)
      {
        return _CUDA_VSTD::make_pair(__first1, __first1 + __len2);
      }
      ++__m1; // no need to check range on __m1 because __s guarantees we have enough source
      if (!__pred(*__m1, *__m2))
      {
        ++__first1;
        break;
      }
    }
  }
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  _ForwardIterator1
  search(_ForwardIterator1 __first1,
         _ForwardIterator1 __last1,
         _ForwardIterator2 __first2,
         _ForwardIterator2 __last2,
         _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__search<__add_lvalue_reference_t<_BinaryPredicate>>(
           __first1,
           __last1,
           __first2,
           __last2,
           __pred,
           typename iterator_traits<_ForwardIterator1>::iterator_category(),
           typename iterator_traits<_ForwardIterator2>::iterator_category())
    .first;
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  _ForwardIterator1
  search(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
  return _CUDA_VSTD::search(__first1, __last1, __first2, __last2, __equal_to{});
}

#if _LIBCUDACXX_STD_VER > 14
template <class _ForwardIterator, class _Searcher>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  _ForwardIterator
  search(_ForwardIterator __f, _ForwardIterator __l, const _Searcher& __s)
{
  return __s(__f, __l).first;
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SEARCH_H
