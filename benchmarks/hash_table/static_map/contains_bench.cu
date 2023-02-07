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

#include "commons.hpp"

#include <cuco/static_map.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>

/**
 * @brief A benchmark evaluating multi-value `contains` performance:
 * - Total number of insertions: 100'000'000
 */
template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_map_contains(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  std::size_t const size = num_keys / occupancy;

  std::vector<Key> h_keys(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);

  generate_keys<Dist, Multiplicity, Key>(h_keys.begin(), h_keys.end());

  for (auto i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  generate_probe_keys<Key>(matching_rate, h_keys.begin(), h_keys.end());

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  cuco::static_map<Key, Value> map{size, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
  map.insert(d_pairs.begin(), d_pairs.end());
  CUCO_CUDA_TRY(cudaStreamSynchronize(nullptr));

  state.add_element_count(num_keys, "NumInputs");
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    map.contains(d_keys.begin(),
                 d_keys.end(),
                 thrust::make_discard_iterator(),
                 cuco::murmurhash3_32<Key>{},
                 thrust::equal_to<Key>{},
                 launch.get_stream());
  });
}

template <typename Key, typename Value, dist_type Dist, nvbench::int32_t Multiplicity>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_map_contains(
  nvbench::state& state,
  nvbench::type_list<Key, Value, nvbench::enum_type<Dist>, nvbench::enum_type<Multiplicity>>)
{
  state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(nvbench_static_map_contains,
                    NVBENCH_TYPE_AXES(KEY_LIST,
                                      VALUE_LIST,
                                      nvbench::enum_type_list<DEFAULT_DISTRIBUTION>,
                                      MULTIPLICITY_LIST))
  .set_name("static_map_contains_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                          // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                          // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {DEFAULT_N})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {DEFAULT_OCCUPANCY})
  .add_float64_axis("MatchingRate", {DEFAULT_MATCHING_RATE});

NVBENCH_BENCH_TYPES(nvbench_static_map_contains,
                    NVBENCH_TYPE_AXES(KEY_LIST,
                                      VALUE_LIST,
                                      DISTRIBUTION_LIST,
                                      nvbench::enum_type_list<DEFAULT_MULTIPLICITY>))
  .set_name("static_map_contains_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                          // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                          // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {DEFAULT_N})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", DEFAULT_OCCUPANCY_RANGE)
  .add_float64_axis("MatchingRate", {DEFAULT_MATCHING_RATE});

NVBENCH_BENCH_TYPES(nvbench_static_map_contains,
                    NVBENCH_TYPE_AXES(KEY_LIST,
                                      VALUE_LIST,
                                      DISTRIBUTION_LIST,
                                      nvbench::enum_type_list<DEFAULT_MULTIPLICITY>))
  .set_name("static_map_contains_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution", "Multiplicity"})
  .set_timeout(100)                          // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                          // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {DEFAULT_N})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {DEFAULT_OCCUPANCY})
  .add_float64_axis("MatchingRate", DEFAULT_MATCHING_RATE_RANGE);
