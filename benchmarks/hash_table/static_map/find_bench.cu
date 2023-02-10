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

#include <defaults.hpp>
#include <key_generator.hpp>

#include <cuco/static_map.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>

namespace cuco::benchmark {
/**
 * @brief A benchmark evaluating `find` performance:
 * - Total number of insertions: 100'000'000
 */
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> static_map_find(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate<Dist>(state, thrust::device, keys.begin(), keys.end());

  auto pairs_begin = thrust::make_transform_iterator(
    keys.begin(), [] __device__(auto i) { return cuco::pair_type<Key, Value>(i, i); });

  cuco::static_map<Key, Value> map{size, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  map.insert(pairs_begin, pairs_begin + num_keys);
  CUCO_CUDA_TRY(cudaStreamSynchronize(nullptr));

  gen.dropout(thrust::device, keys.begin(), keys.end(), matching_rate);

  state.add_element_count(num_keys, "NumInputs");
  state.set_global_memory_rw_bytes(num_keys * sizeof(cuco::pair_type<Key, Value>));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.find(keys.begin(),
             keys.end(),
             thrust::make_discard_iterator(),
             cuco::murmurhash3_32<Key>{},
             thrust::equal_to<Key>{},
             launch.get_stream());
  });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_find(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}

using namespace defaults;

NVBENCH_BENCH_TYPES(static_map_find,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::unique>))
  .set_name("static_map_find_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)                  // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(MAX_NOISE)          // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {N})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", OCCUPANCY_RANGE)
  .add_float64_axis("MatchingRate", {MATCHING_RATE});

NVBENCH_BENCH_TYPES(static_map_find,
                    NVBENCH_TYPE_AXES(KEY_TYPE_RANGE,
                                      VALUE_TYPE_RANGE,
                                      nvbench::type_list<dist_type::unique>))
  .set_name("static_map_find_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)                  // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(MAX_NOISE)          // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {N})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {OCCUPANCY})
  .add_float64_axis("MatchingRate", MATCHING_RATE_RANGE);
}  // namespace cuco::benchmark
