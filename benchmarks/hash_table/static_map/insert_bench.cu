/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <utils.hpp>

#include <cuco/static_map.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <cuda/std/array>

using namespace cuco::benchmark;
using namespace cuco::utility;

template <int32_t Bytes>
struct bytes {
  using word_type = int32_t;
  static_assert(Bytes % sizeof(word_type) == 0);

  static constexpr auto words = Bytes / sizeof(word_type);

  // Default constructor initializing all elements to 0
  __host__ __device__ constexpr bytes() noexcept : data_{} {}

  // Constructor with seed to initialize all elements
  __host__ __device__ constexpr bytes(word_type seed) noexcept
    : data_(init_array(seed, cuda::std::make_index_sequence<words>{}))
  {
  }

  // Comparison operator
  __host__ __device__ constexpr bool operator==(const bytes& other) const
  {
    return cuda::std::equal(data_.begin(), data_.end(), other.data_.begin());
  }

 private:
  template <std::size_t... I>
  __host__ __device__ static constexpr cuda::std::array<word_type, words> init_array(
    word_type seed, cuda::std::index_sequence<I...>)
  {
    // Use seed and expand it Words times using the pack expansion trick with I...
    return {{((void)I, seed)...}};
  }

  cuda::std::array<word_type, words> data_;
};

/**
 * @brief A benchmark evaluating `cuco::static_map::insert` performance
 */
template <typename Key, typename Value, typename Dist>
void static_map_insert(nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  using pair_type = cuco::pair<Key, Value>;

  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  thrust::device_vector<pair_type> pairs(num_keys);
  thrust::transform(keys.begin(), keys.end(), pairs.begin(), [] __device__(Key const& key) {
    return pair_type(key, {});
  });

  state.add_element_count(num_keys);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    auto map = cuco::static_map{size,
                                cuco::empty_key<Key>{-1},
                                cuco::empty_value<Value>{-1},
                                {},
                                {},
                                {},
                                {},
                                {},
                                {launch.get_stream()}};

    timer.start();
    map.insert_async(pairs.begin(), pairs.end(), {launch.get_stream()});
    timer.stop();
  });
}

/*
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_insert(
  nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
  state.skip("Key should be the same type as Value.");
}
*/

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::uniform>))
  .set_name("static_map_insert_uniform_multiplicity")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_int64_axis("Multiplicity", defaults::MULTIPLICITY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_map_insert_unique_occupancy")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                      defaults::VALUE_TYPE_RANGE,
                                      nvbench::type_list<distribution::gaussian>))
  .set_name("static_map_insert_gaussian_skew")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Skew", defaults::SKEW_RANGE);

NVBENCH_BENCH_TYPES(static_map_insert,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<bytes<128>, bytes<256>, bytes<512>>,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_map_insert_unique_occupancy_large_payload")
  .set_type_axes_names({"Key", "Value", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE)
  .add_int64_axis("NumInputs", {100'000});
