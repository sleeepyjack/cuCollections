/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utils.hpp>

#include <cuco/static_multimap.cuh>

#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <catch2/catch_template_test_macros.hpp>

template <typename Map>
__inline__ void test_multiplicity_two(Map& map, std::size_t num_items)
{
  using Key   = typename Map::key_type;
  using Value = typename Map::mapped_type;

  thrust::device_vector<Key> d_keys(num_items / 2);
  thrust::device_vector<cuco::pair<Key, Value>> d_pairs(num_items);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  // multiplicity = 2
  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(num_items),
                    d_pairs.begin(),
                    [] __device__(auto i) {
                      return cuco::pair<Key, Value>{i / 2, i};
                    });

  // The following line is somehow causing the segfault, even though d_contained is never used in
  // the code. If you comment out this line, the nvcc segfault goes away.
  thrust::device_vector<bool> d_contained(num_items / 2);

  map.insert(d_pairs.begin(), d_pairs.end());
}

TEMPLATE_TEST_CASE_SIG(
  "Multiplicity equals two",
  "",
  ((typename Key, typename Value, cuco::test::probe_sequence Probe), Key, Value, Probe),
  (int32_t, int32_t, cuco::test::probe_sequence::linear_probing))
{
  constexpr std::size_t num_items{4};

  using probe = std::conditional_t<Probe == cuco::test::probe_sequence::linear_probing,
                                   cuco::linear_probing<1, cuco::default_hash_function<Key>>,
                                   cuco::double_hashing<8, cuco::default_hash_function<Key>>>;

  cuco::static_multimap<Key, Value, cuda::thread_scope_device, cuco::cuda_allocator<char>, probe>
    map{5, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  test_multiplicity_two(map, num_items);
}
