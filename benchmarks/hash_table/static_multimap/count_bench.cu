/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cuco/static_multimap.cuh>

#include <nvbench/nvbench.cuh>

#include <key_generator.hpp>
#include <utils.cuh>

#include <thrust/device_vector.h>

/**
 * @brief A benchmark evaluating multi-value `count` performance:
 * - Total number of insertions: 100'000'000
 * - CG size: 8
 */
template <typename Key, typename Value, dist_type Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state, nvbench::type_list<Key, Value, nvbench::enum_type<Dist>>)
{
  auto const num_keys      = state.get_int64("NumInputs");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const matching_rate = state.get_float64("MatchingRate");

  key_generator<Dist> gen;

  switch (Dist) {
    case dist_type::UNIQUE: break;
    case dist_type::GAUSSIAN:
      gen = key_generator<Dist>(state.get_float64_or_default("Deviation", 0.2));
      break;
    case dist_type::UNIFORM:
      gen = key_generator<Dist>(state.get_int64_or_default("Multiplicity", 8));
      break;
  }

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> pairs(num_keys);

  state.add_element_count(num_keys, "NumKeys");

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      gen.generate(thrust::cuda::par.on(launch.get_stream()), keys.begin(), keys.end());

      thrust::transform(thrust::cuda::par.on(launch.get_stream()),
                        keys.cbegin(),
                        keys.cend(),
                        pairs.begin(),
                        make_cuco_pair<Key, Value>{});

      cuco::static_multimap<Key, Value> map{size,
                                            cuco::sentinel::empty_key<Key>{-1},
                                            cuco::sentinel::empty_value<Value>{-1},
                                            launch.get_stream()};
      map.insert(pairs.cbegin(), pairs.cend(), launch.get_stream());

      gen.dropout(
        thrust::cuda::par.on(launch.get_stream()), keys.begin(), keys.end(), matching_rate);

      timer.start();
      auto count = map.count(keys.cbegin(), keys.cend(), launch.get_stream());
      timer.stop();
    });
}

template <typename Key, typename Value, dist_type Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> nvbench_static_multimap_count(
  nvbench::state& state, nvbench::type_list<Key, Value, nvbench::enum_type<Dist>>)
{
  state.skip("Key should be the same type as Value.");
}

using key_type   = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int32_t, nvbench::int64_t>;
using d_type     = nvbench::enum_type_list<dist_type::GAUSSIAN, dist_type::UNIFORM>;

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count,
                    NVBENCH_TYPE_AXES(key_type,
                                      value_type,
                                      nvbench::enum_type_list<dist_type::UNIFORM>))
  .set_name("staic_multimap_count_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.5})
  .add_int64_axis("Multiplicity", {1, 2, 4, 8, 16, 32, 64, 128, 256});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count, NVBENCH_TYPE_AXES(key_type, value_type, d_type))
  .set_name("staic_multimap_count_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1))
  .add_float64_axis("MatchingRate", {0.5});

NVBENCH_BENCH_TYPES(nvbench_static_multimap_count, NVBENCH_TYPE_AXES(key_type, value_type, d_type))
  .set_name("staic_multimap_count_matching_rate")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_timeout(100)                            // Custom timeout: 100 s. Default is 15 s.
  .set_max_noise(3)                            // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("NumInputs", {100'000'000})  // Total number of key/value pairs: 100'000'000
  .add_float64_axis("Occupancy", {0.8})
  .add_float64_axis("MatchingRate", {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1});
