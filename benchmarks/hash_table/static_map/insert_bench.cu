/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuco/static_map.cuh>

#include <nvbench/nvbench.cuh>

#include <defaults.cuh>
#include <key_generator.hpp>
#include <utils.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

/**
 * @brief A benchmark evaluating `cuco::static_map::insert` performance.
 */
template <typename Pair, dist_type Dist>
void insert_bench(nvbench::state& state, nvbench::type_list<Pair, nvbench::enum_type<Dist>>)
{
  using key_type      = typename Pair::first_type;
  using mapped_type   = typename Pair::second_type;
  using map_type      = typename cuco::static_map<key_type, mapped_type>;
  using generator_typ = key_generator<Dist>;

  auto const num_keys    = state.get_int64_or_default("NumInputs", defaults::num_keys);
  auto const occupancy   = state.get_float64_or_default("Occupancy", defaults::occupancy);
  std::size_t const size = num_keys / occupancy;

  generator_typ gen;
  switch (Dist) {
    case dist_type::UNIQUE: break;
    case dist_type::GAUSSIAN:
      gen = generator_typ(state.get_float64_or_default("Deviation", defaults::deviation));
      break;
    case dist_type::UNIFORM:
      gen = generator_typ(state.get_int64_or_default("Multiplicity", defaults::multiplicity));
      break;
  }

  thrust::device_vector<key_type> keys(num_keys);
  thrust::device_vector<Pair> pairs(num_keys);

  state.add_element_count(num_keys);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               gen.generate(thrust::cuda::par.on(launch.get_stream()), keys.begin(), keys.end());

               thrust::transform(thrust::cuda::par.on(launch.get_stream()),
                                 keys.cbegin(),
                                 keys.cend(),
                                 pairs.begin(),
                                 make_cuco_pair<key_type, mapped_type>{});

               map_type map{size,
                            cuco::sentinel::empty_key<key_type>{-1},
                            cuco::sentinel::empty_value<mapped_type>{-1},
                            cuco::cuda_allocator<char>{},
                            launch.get_stream()};

               timer.start();
               map.insert(pairs.cbegin(),
                          pairs.cend(),
                          cuco::detail::MurmurHash3_32<key_type>{},
                          thrust::equal_to<key_type>{},
                          launch.get_stream());
               timer.stop();
             });
}

NVBENCH_BENCH_TYPES(insert_bench,
                    NVBENCH_TYPE_AXES(defaults::pair_type_list,
                                      nvbench::enum_type_list<dist_type::UNIQUE>))
  .set_name("cuco::static_map::insert [occupancy]")
  .set_type_axes_names({"Pair", "Distribution"})
  .set_max_noise(defaults::max_noise)
  .add_float64_axis("Occupancy", nvbench::range(0.1, 0.9, 0.1));

NVBENCH_BENCH_TYPES(insert_bench,
                    NVBENCH_TYPE_AXES(defaults::pair_type_list,
                                      nvbench::enum_type_list<dist_type::UNIFORM>))
  .set_name("cuco::static_map::insert [multiplicity]")
  .set_type_axes_names({"Pair", "Distribution"})
  .set_max_noise(defaults::max_noise)
  .add_float64_axis("Multiplicity", {1, 2, 4, 8, 16, 32, 64, 128, 256});

NVBENCH_BENCH_TYPES(insert_bench,
                    NVBENCH_TYPE_AXES(defaults::pair_type_list,
                                      nvbench::enum_type_list<dist_type::GAUSSIAN>))
  .set_name("cuco::static_map::insert [skew]")
  .set_type_axes_names({"Pair", "Distribution"})
  .set_max_noise(defaults::max_noise)
  .add_float64_axis("Deviation", nvbench::range(0.1, 1.0, 0.1));
