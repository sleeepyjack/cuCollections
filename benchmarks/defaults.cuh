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
#include <nvbench/nvbench.cuh>

#include <cuco/detail/pair.cuh>
#include <key_generator.hpp>

struct defaults {
  using pair_type_list = nvbench::type_list<cuco::pair<nvbench::int32_t, nvbench::int32_t>,
                                            cuco::pair<nvbench::int64_t, nvbench::int64_t>>;
  using dist_type_list =
    nvbench::enum_type_list<dist_type::UNIQUE, dist_type::GAUSSIAN, dist_type::UNIFORM>;
  static constexpr nvbench::int64_t num_keys     = 100'000'000;
  static constexpr nvbench::float64_t occupancy  = 0.8;
  static constexpr nvbench::int64_t multiplicity = 8;
  static constexpr nvbench::float64_t deviation  = 0.2;
  static constexpr nvbench::float64_t max_noise  = 3.0;
  static constexpr nvbench::float64_t match_rate = 1.0;
};