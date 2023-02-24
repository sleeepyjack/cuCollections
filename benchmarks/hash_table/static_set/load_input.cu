/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuco/static_set.cuh>
#include <cuco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

using namespace cuco::benchmark;
using namespace cuco::utility;

// simple call to set.insert()
struct baseline {
  template <typename Set, typename InputIt>
  static void insert(Set& set, InputIt begin, InputIt end, cudaStream_t stream = nullptr)
  {
    set.insert(begin, end, stream);
  }
};

template <int32_t BlockSize, typename InputIterator, typename Reference>
__global__ void insert_repro(InputIterator first, cuco::detail::index_type n, Reference set_ref)
{
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = BlockSize * blockIdx.x + threadIdx.x;

  while (idx < n) {
    typename Reference::value_type const insert_pair{*(first + idx)};
    set_ref.insert(insert_pair);
    idx += loop_stride;
  }
}

template <int32_t CGSize, int32_t BlockSize, typename InputIterator, typename Reference>
__global__ void insert_repro(InputIterator first, cuco::detail::index_type n, Reference set_ref)
{
  namespace cg = cooperative_groups;

  auto tile                                  = cg::tiled_partition<CGSize>(cg::this_thread_block());
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize / CGSize;
  cuco::detail::index_type idx               = (BlockSize * blockIdx.x + threadIdx.x) / CGSize;

  while (idx < n) {
    typename Reference::value_type const insert_pair{*(first + idx)};
    set_ref.insert(tile, insert_pair);
    idx += loop_stride;
  }
}

// same as baseline, but call kernel explicitly; sanity check
struct repro {
  template <typename Set, typename InputIt>
  static void insert(Set& set, InputIt first, InputIt last, cudaStream_t stream = nullptr)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = (Set::cg_size * num_keys +
                            cuco::experimental::detail::CUCO_DEFAULT_STRIDE *
                              cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE -
                            1) /
                           (cuco::experimental::detail::CUCO_DEFAULT_STRIDE *
                            cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE);

    if constexpr (Set::cg_size == 1) {
      insert_repro<cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE>
        <<<grid_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
          first, num_keys, set.ref_with(cuco::experimental::op::insert));
    } else {
      insert_repro<Set::cg_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE>
        <<<grid_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
          first, num_keys, set.ref_with(cuco::experimental::op::insert));
    }
  }
};

template <int32_t BlockSize, typename InputIterator, typename Reference>
__global__ void insert_from_register(InputIterator first,
                                     cuco::detail::index_type n,
                                     Reference set_ref)
{
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize;
  cuco::detail::index_type idx               = BlockSize * blockIdx.x + threadIdx.x;

  while (idx < n) {
    set_ref.insert(static_cast<typename Reference::key_type>(idx));
    idx += loop_stride;
  }
}

template <int32_t CGSize, int32_t BlockSize, typename InputIterator, typename Reference>
__global__ void insert_from_register(InputIterator first,
                                     cuco::detail::index_type n,
                                     Reference set_ref)
{
  namespace cg = cooperative_groups;

  auto tile                                  = cg::tiled_partition<CGSize>(cg::this_thread_block());
  cuco::detail::index_type const loop_stride = gridDim.x * BlockSize / CGSize;
  cuco::detail::index_type idx               = (BlockSize * blockIdx.x + threadIdx.x) / CGSize;

  while (idx < n) {
    set_ref.insert(tile, static_cast<typename Reference::key_type>(idx));
    idx += loop_stride;
  }
}

// don't load input from global memory, but a register
struct from_register {
  template <typename Set, typename InputIt>
  static void insert(Set& set, InputIt first, InputIt last, cudaStream_t stream = nullptr)
  {
    auto const num_keys = cuco::detail::distance(first, last);
    if (num_keys == 0) { return; }

    auto const grid_size = (Set::cg_size * num_keys +
                            cuco::experimental::detail::CUCO_DEFAULT_STRIDE *
                              cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE -
                            1) /
                           (cuco::experimental::detail::CUCO_DEFAULT_STRIDE *
                            cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE);

    if constexpr (Set::cg_size == 1) {
      insert_from_register<cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE>
        <<<grid_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
          first, num_keys, set.ref_with(cuco::experimental::op::insert));
    } else {
      insert_from_register<Set::cg_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE>
        <<<grid_size, cuco::experimental::detail::CUCO_DEFAULT_BLOCK_SIZE, 0, stream>>>(
          first, num_keys, set.ref_with(cuco::experimental::op::insert));
    }
  }
};

/**
 * @brief A benchmark evaluating different data load patterns
 */
template <typename Runner, typename Key, typename Dist>
void static_set_load_input(nvbench::state& state, nvbench::type_list<Runner, Key, Dist>)
{
  auto const num_keys  = state.get_int64_or_default("NumInputs", defaults::N);
  auto const occupancy = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);

  key_generator gen;
  gen.generate(dist_from_state<Dist>(state), keys.begin(), keys.end());

  state.add_element_count(num_keys);

  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               cuco::experimental::static_set<Key> set{
                 size, cuco::empty_key<Key>{-1}, {}, {}, {}, launch.get_stream()};

               timer.start();
               Runner::insert(set, keys.begin(), keys.end(), launch.get_stream());
               timer.stop();
             });
}

using RUNNER_RANGE = nvbench::type_list<baseline, repro, from_register>;

NVBENCH_BENCH_TYPES(static_set_load_input,
                    NVBENCH_TYPE_AXES(RUNNER_RANGE,
                                      defaults::KEY_TYPE_RANGE,
                                      nvbench::type_list<distribution::unique>))
  .set_name("static_set_load_input_unique_occupancy")
  .set_type_axes_names({"Runner", "Key", "Distribution"})
  .set_max_noise(defaults::MAX_NOISE)
  .add_float64_axis("Occupancy", defaults::OCCUPANCY_RANGE);
