#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/logical.h>

#include <unittest/unittest.h>

template <class Vector>
void TestAllOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), ::cuda::std::identity{}), true);

  v[1] = T{0};

  ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), ::cuda::std::identity{}), false);

  ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}), true);
  ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}), true);
  ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}), false);
  ASSERT_EQUAL(thrust::all_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}), false);
}
DECLARE_VECTOR_UNITTEST(TestAllOf);

template <class InputIterator, class Predicate>
bool all_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

void TestAllOfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::all_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestAllOfDispatchExplicit);

template <class InputIterator, class Predicate>
bool all_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

void TestAllOfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::all_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAllOfDispatchImplicit);

template <class Vector>
void TestAnyOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), ::cuda::std::identity{}), true);

  v[1] = 0;

  ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), ::cuda::std::identity{}), true);

  ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}), false);
  ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}), true);
  ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}), true);
  ASSERT_EQUAL(thrust::any_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}), false);
}
DECLARE_VECTOR_UNITTEST(TestAnyOf);

template <class InputIterator, class Predicate>
bool any_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

void TestAnyOfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::any_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestAnyOfDispatchExplicit);

template <class InputIterator, class Predicate>
bool any_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

void TestAnyOfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::any_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAnyOfDispatchImplicit);

template <class Vector>
void TestNoneOf()
{
  using T = typename Vector::value_type;

  Vector v(3, T{1});

  ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), ::cuda::std::identity{}), false);

  v[1] = 0;

  ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), ::cuda::std::identity{}), false);

  ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 0, ::cuda::std::identity{}), true);
  ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 1, ::cuda::std::identity{}), false);
  ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 2, ::cuda::std::identity{}), false);
  ASSERT_EQUAL(thrust::none_of(v.begin() + 1, v.begin() + 2, ::cuda::std::identity{}), true);
}
DECLARE_VECTOR_UNITTEST(TestNoneOf);

template <class InputIterator, class Predicate>
bool none_of(my_system& system, InputIterator, InputIterator, Predicate)
{
  system.validate_dispatch();
  return false;
}

void TestNoneOfDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::none_of(sys, vec.begin(), vec.end(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestNoneOfDispatchExplicit);

template <class InputIterator, class Predicate>
bool none_of(my_tag, InputIterator first, InputIterator, Predicate)
{
  *first = 13;
  return false;
}

void TestNoneOfDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::none_of(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestNoneOfDispatchImplicit);
