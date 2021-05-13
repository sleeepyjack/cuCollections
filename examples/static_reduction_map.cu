/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <iostream>
#include <limits>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuco/static_reduction_map.cuh>

/**
 * @file host_bulk_example.cu
 * @brief Demonstrates usage of the static_map "bulk" host APIs.
 *
 * The bulk APIs are only invocable from the host and are used for doing operations like insert or
 * find on a set of keys.
 *
 */

int main(void)
{
  using Key   = int;
  using Value = int;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key const empty_key_sentinel = -1;

  // Number of key/value pairs to be inserted
  std::size_t num_keys = 50'000;

  // Compute capacity based on a 50% load factor
  auto const load_factor     = 0.5;
  std::size_t const capacity = std::ceil(num_keys / load_factor);

  // Constructs a map each key with "capacity" slots using -1 as the
  // empty key sentinel. The initial payload value for empty slots is determined by the identity of
  // the reduction operation. By using the `reduce_add` operation, all values associated with a
  // given key will be summed.
  cuco::static_reduction_map<cuco::reduce_add<Value>, Key, Value> map{capacity, empty_key_sentinel};

  // Create a sequence of random keys in `[0, num_keys/2]`
  thrust::device_vector<Key> insert_keys(num_keys);
  thrust::transform(thrust::device,
                    thrust::make_counting_iterator<std::size_t>(0),
                    thrust::make_counting_iterator(insert_keys.size()),
                    insert_keys.begin(),
                    [=] __device__(auto i) {
                      thrust::default_random_engine rng(i);
                      thrust::uniform_int_distribution dist{std::size_t{0}, num_keys/2};
                      return dist(rng);
                    });

  // Insert each key with a payload of `1` to count the number of times each key was inserted by
  // using the `reduce_add` op
  auto zipped = thrust::make_zip_iterator(
    thrust::make_tuple(insert_keys.begin(), thrust::make_constant_iterator(1)));

  // Inserts all pairs into the map, accumulating the payloads with the `reduce_add` operation
  map.insert(zipped, zipped + insert_keys.size());

  std::cout << "Num unique keys: " << map.get_size() << std::endl;

}