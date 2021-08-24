/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <random>

#include <thrust/device_vector.h>
#include <nvbench/nvbench.cuh>

#include <cuco/bloom_filter.cuh>
#include <cuco/static_multimap.cuh>
#include <key_generator.hpp>

template <typename Key, typename Value>
void nvbench_hash_join(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  cudaCtxResetPersistingL2Cache();

  using map_type = cuco::static_multimap<Key, Value>;

  auto const r_size        = state.get_int64("RSize");
  auto const s_size        = state.get_int64("SSize");
  auto const matching_rate = state.get_float64("MatchingRate");
  auto const multiplicity  = state.get_int64("Multiplicity");
  auto const occupancy     = state.get_float64("Occupancy");

  if (r_size > s_size) {
    state.skip("Large build tables are skipped.");
    return;
  }

  if (r_size * 100 <= s_size) {
    state.skip("Large probe tables are skipped.");
    return;
  }

  std::size_t const capacity = r_size / occupancy;

  std::vector<Key> r_keys_h(r_size);
  std::vector<Key> s_keys_h(s_size);

  generate_join_keys<Key>(r_keys_h.begin(),
                          r_keys_h.end(),
                          s_keys_h.begin(),
                          s_keys_h.end(),
                          matching_rate,
                          multiplicity);

  thrust::device_vector<Key> r_keys_d(r_keys_h);
  thrust::device_vector<Key> s_keys_d(s_keys_h);

  thrust::device_vector<typename map_type::value_type> r_pairs_d(r_size);
  thrust::transform(r_keys_d.begin(), r_keys_d.end(), r_pairs_d.begin(), [] __device__(auto i) {
    return typename map_type::value_type{i, i};
  });

  std::size_t free_mem;
  std::size_t total_mem;

  cudaMemGetInfo(&free_mem, &total_mem);
  std::size_t const max_result_size = (free_mem * 0.8) / sizeof(typename map_type::value_type);

  thrust::device_vector<typename map_type::value_type> result_d(max_result_size);

  std::size_t count_h = 0;
  {
    map_type map{capacity, -1, -1};
    map.insert(r_pairs_d.begin(), r_pairs_d.end());
    count_h = map.count(s_keys_d.begin(), s_keys_d.end());
  }

  if (count_h > max_result_size) {
    state.skip("Result is too large.");
    return;
  }

  state.add_element_count(count_h);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      cuco::static_multimap<Key, Value> map{capacity, -1, -1};

      timer.start();
      map.insert(r_pairs_d.begin(), r_pairs_d.end(), launch.get_stream());
      count_h = map.count(s_keys_d.begin(), s_keys_d.end(), launch.get_stream());
      map.retrieve(s_keys_d.begin(), s_keys_d.end(), result_d.begin(), launch.get_stream());
      timer.stop();
    });
}

template <std::size_t CGSize,
          std::size_t BlockSize,
          typename Pair,
          typename FilterType,
          typename MapType>
__global__ void filtered_insert(Pair* input, std::size_t size, FilterType filter, MapType map)
{
  auto tile = cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
  typename MapType::value_type thread_pair{map.get_empty_key_sentinel(),
                                           map.get_empty_value_sentinel()};

  for (std::size_t tid = BlockSize * blockIdx.x + threadIdx.x; tid < (SDIV(size, CGSize) * CGSize);
       tid += gridDim.x * BlockSize) {
    if (tid < size) {
      thread_pair = input[tid];
      filter.insert(thread_pair.first);
    }

#pragma unroll CGSize
    for (std::size_t lane = 0; lane < CGSize; ++lane) {
      auto tile_pair = tile.shfl(thread_pair, lane);
      if (tile_pair.first != map.get_empty_key_sentinel()) { map.insert(tile, tile_pair); }
    }
  }
}

template <std::size_t CGSize,
          std::size_t BlockSize,
          typename Key,
          typename Counter,
          typename FilterType,
          typename MapType>
__global__ void filtered_count(
  Key* input, std::size_t size, Counter* num_matches, FilterType filter, MapType map)
{
  auto tile = cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
  typename MapType::key_type thread_key;

  typedef cub::BlockReduce<std::size_t, BlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_matches = 0;

  bool contained;
  std::uint32_t contained_mask;

  for (std::size_t tid = BlockSize * blockIdx.x + threadIdx.x; tid < (SDIV(size, CGSize) * CGSize);
       tid += gridDim.x * BlockSize) {
    thread_key = (tid < size) ? input[tid] : map.get_empty_key_sentinel();
    contained  = (tid < size) ? filter.contains(thread_key) : false;

    contained_mask = tile.ballot(contained);

    while (contained_mask) {
      auto const leader   = __ffs(contained_mask) - 1;
      auto const tile_key = tile.shfl(thread_key, leader);

      map.count(tile, tile_key, thread_num_matches);

      contained_mask ^= 1UL << leader;
    }
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_matches = BlockReduce(temp_storage).Sum(thread_num_matches);
  if (threadIdx.x == 0) {
    num_matches->fetch_add(block_num_matches, cuda::std::memory_order_relaxed);
  }
}

template <std::size_t CGSize,
          std::size_t BlockSize,
          typename Key,
          typename Pair,
          typename Counter,
          typename FilterType,
          typename MapType>
__global__ void filtered_retrieve(
  Key* input, std::size_t size, Pair* output, Counter* num_matches, FilterType filter, MapType map)
{
  namespace cg = cooperative_groups;

  constexpr uint32_t buffer_size = CGSize * 3u;

  auto tile                  = cg::tiled_partition<CGSize>(cg::this_thread_block());
  auto tid                   = BlockSize * blockIdx.x + threadIdx.x;
  constexpr uint32_t num_cgs = BlockSize / CGSize;
  const uint32_t cg_id       = threadIdx.x / CGSize;
  const uint32_t lane_id     = tile.thread_rank();

  __shared__ Pair output_buffer[num_cgs][buffer_size];
  // TODO: replace this with shared memory cuda::atomic variables once the dynamiic initialization
  // warning issue is solved __shared__ atomicT toto_countter[num_warps];
  __shared__ uint32_t cg_counter[num_cgs];

  if (lane_id == 0) { cg_counter[cg_id] = 0; }

  while (tile.any(tid < size)) {
    bool active_lane_flag = tid < size;
    auto thread_key       = (active_lane_flag) ? input[tid] : map.get_empty_key_sentinel();
    active_lane_flag      = (active_lane_flag) ? filter.contains(thread_key) : false;
    auto active_lane_mask = tile.ballot(active_lane_flag);

    while (active_lane_mask) {
      auto const leader   = __ffs(active_lane_mask) - 1;
      auto const tile_key = tile.shfl(thread_key, leader);

      map.retrieve<CGSize, buffer_size>(
        tile, tile_key, &cg_counter[cg_id], output_buffer[cg_id], num_matches, output);

      active_lane_mask ^= 1UL << leader;
    }
    tid += gridDim.x * BlockSize;
  }

  // Final flush of output buffer
  if (cg_counter[cg_id] > 0) {
    map.flush_output_buffer(tile, cg_counter[cg_id], output_buffer[cg_id], num_matches, output);
  }
}

template <typename Kernel>
auto get_grid_size(std::size_t block_size, Kernel kernel)
{
  int grid_size{-1};
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, kernel, block_size, 0);
  int dev_id{-1};
  cudaGetDevice(&dev_id);
  int num_sms{-1};
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);
  grid_size *= num_sms;
  return grid_size;
}

template <typename Key, typename Value>
void nvbench_filtered_hash_join(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  cudaCtxResetPersistingL2Cache();

  using map_type              = cuco::static_multimap<Key, Value>;
  using map_mutable_view_type = typename map_type::device_mutable_view;
  using map_view_type         = typename map_type::device_view;

  using filter_type              = cuco::bloom_filter<Key>;
  using filter_mutable_view_type = typename filter_type::device_mutable_view;
  using filter_view_type         = typename filter_type::device_view;

  using pair_type       = typename map_type::value_type;
  using atomic_ctr_type = typename map_type::atomic_ctr_type;

  auto const r_size        = state.get_int64("RSize");
  auto const s_size        = state.get_int64("SSize");
  auto const matching_rate = state.get_float64("MatchingRate");
  auto const multiplicity  = state.get_int64("Multiplicity");
  auto const occupancy     = state.get_float64("Occupancy");
  auto const num_hashes    = state.get_int64("NumHashes");
  auto const l2_hit_rate   = state.get_float64("L2HitRate");

  if (r_size > s_size) {
    state.skip("Large build tables are skipped.");
    return;
  }

  if (r_size * 100 <= s_size) {
    state.skip("Large probe tables are skipped.");
    return;
  }

  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);

  std::size_t const filter_bits  = prop.persistingL2CacheMaxSize * CHAR_BIT;
  std::size_t const map_capacity = r_size / occupancy;

  std::vector<Key> r_keys_h(r_size);
  std::vector<Key> s_keys_h(s_size);

  generate_join_keys<Key>(r_keys_h.begin(),
                          r_keys_h.end(),
                          s_keys_h.begin(),
                          s_keys_h.end(),
                          matching_rate,
                          multiplicity);

  thrust::device_vector<Key> r_keys_d(r_keys_h);
  thrust::device_vector<Key> s_keys_d(s_keys_h);

  thrust::device_vector<pair_type> r_pairs_d(r_size);
  thrust::transform(r_keys_d.begin(), r_keys_d.end(), r_pairs_d.begin(), [] __device__(auto i) {
    return pair_type{i, i};
  });

  float filter_fp;
  {  // estimate false positive rate of bloom filter
    thrust::device_vector<Key> tp_d(r_keys_h);

    filter_type filter(filter_bits, num_hashes);
    filter.insert(tp_d.begin(), tp_d.end());

    thrust::device_vector<Key> tn_d(r_size);
    thrust::sequence(thrust::device, tn_d.begin(), tn_d.end(), r_size);

    thrust::device_vector<bool> contained_d(r_size, false);

    filter.contains(tn_d.begin(), tn_d.end(), contained_d.begin());

    filter_fp = thrust::count(thrust::device, contained_d.begin(), contained_d.end(), true);
  }

  auto& summ = state.add_summary("Filter False-Positive Rate");
  summ.set_string("hint", "FilterFPR");
  summ.set_string("short_name", "FilterFPR");
  summ.set_string("description", "False-positive rate of the bloom filter.");
  summ.set_float64("value", float(filter_fp) / r_size);

  std::size_t count_h         = 0;
  std::size_t max_result_size = 0;
  {  // estimate result size
    map_type map(map_capacity, -1, -1);
    filter_type filter(filter_bits, num_hashes);

    std::size_t free_mem;
    std::size_t total_mem;

    cudaMemGetInfo(&free_mem, &total_mem);
    max_result_size = (free_mem * 0.8) / sizeof(pair_type);

    map.insert(r_pairs_d.begin(), r_pairs_d.end());
    count_h = map.count(s_keys_d.begin(), s_keys_d.end());
  }

  if (count_h > max_result_size) {
    state.skip("Result is too large.");
    return;
  }

  thrust::device_vector<pair_type> result_d(count_h);

  atomic_ctr_type* count_d = nullptr;
  CUCO_CUDA_TRY(cudaMalloc(&count_d, sizeof(atomic_ctr_type)));

  // determine optimal launch parameters
  std::size_t constexpr cg_size = 8;

  std::size_t constexpr insert_block_size   = 128;
  std::size_t constexpr count_block_size    = 128;
  std::size_t constexpr retrieve_block_size = 128;

  std::size_t const insert_grid_size = get_grid_size(insert_block_size,
                                                     filtered_insert<cg_size,
                                                                     insert_block_size,
                                                                     pair_type,
                                                                     filter_mutable_view_type,
                                                                     map_mutable_view_type>);
  std::size_t const count_grid_size  = get_grid_size(count_block_size,
                                                    filtered_count<cg_size,
                                                                   count_block_size,
                                                                   Key,
                                                                   atomic_ctr_type,
                                                                   filter_view_type,
                                                                   map_view_type>);

  std::size_t const retrieve_grid_size = get_grid_size(retrieve_block_size,
                                                       filtered_retrieve<cg_size,
                                                                         retrieve_block_size,
                                                                         Key,
                                                                         pair_type,
                                                                         atomic_ctr_type,
                                                                         filter_view_type,
                                                                         map_view_type>);

  state.add_element_count(count_h);

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      map_type map(map_capacity, -1, -1);
      auto mutable_map_view = map.get_device_mutable_view();
      auto map_view         = map.get_device_view();

      filter_type filter(filter_bits, num_hashes);
      auto mutable_filter_view = filter.get_device_mutable_view();
      auto filter_view         = filter.get_device_view();

      cudaStreamAttrValue stream_attribute;
      stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(filter.get_slots());
      stream_attribute.accessPolicyWindow.num_bytes = filter.get_num_bits() / CHAR_BIT;
      stream_attribute.accessPolicyWindow.hitRatio  = l2_hit_rate;
      stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
      stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(
        launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

      timer.start();

      // map.insert(r_pairs_d.begin(), r_pairs_d.end(), launch.get_stream());
      filtered_insert<cg_size, insert_block_size>
        <<<insert_grid_size, insert_block_size, 0, launch.get_stream()>>>(
          r_pairs_d.data().get(), r_size, mutable_filter_view, mutable_map_view);

      // count_h = map.count(s_keys_d.begin(), s_keys_d.end(), launch.get_stream());
      cudaMemsetAsync(count_d, 0, sizeof(atomic_ctr_type), launch.get_stream());
      filtered_count<cg_size, count_block_size>
        <<<count_grid_size, count_block_size, 0, launch.get_stream()>>>(
          s_keys_d.data().get(), s_size, count_d, filter_view, map_view);
      cudaMemcpyAsync(
        &count_h, count_d, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, launch.get_stream());

      // map.retrieve(s_keys_d.begin(), s_keys_d.end(), result_d.begin(), launch.get_stream());
      cudaMemsetAsync(count_d, 0, sizeof(atomic_ctr_type), launch.get_stream());
      filtered_retrieve<cg_size, retrieve_block_size>
        <<<retrieve_grid_size, retrieve_block_size, 0, launch.get_stream()>>>(
          s_keys_d.data().get(), s_size, result_d.data().get(), count_d, filter_view, map_view);
      cudaMemcpyAsync(
        &count_h, count_d, sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, launch.get_stream());

      timer.stop();

      stream_attribute.accessPolicyWindow.num_bytes = 0;
      cudaStreamSetAttribute(
        launch.get_stream(), cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

      cudaCtxResetPersistingL2Cache();
    });
  CUCO_CUDA_TRY(cudaFree(count_d));
}

using key_type   = nvbench::type_list<nvbench::int64_t>;
using value_type = nvbench::type_list<nvbench::int64_t>;

NVBENCH_BENCH_TYPES(nvbench_filtered_hash_join, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_name("nvbench_filtered_hash_join")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("RSize", {50'000'000})
  .add_int64_axis("SSize", {50'000'000, 100'000'000, 200'000'000})
  .add_float64_axis("MatchingRate", {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.})
  .add_int64_axis("Multiplicity", {1, 8, 16, 32, 64})
  .add_float64_axis("Occupancy", {0.8})
  .add_int64_axis("NumHashes", {2})
  .add_float64_axis("L2HitRate", {0.6});

NVBENCH_BENCH_TYPES(nvbench_hash_join, NVBENCH_TYPE_AXES(key_type, value_type))
  .set_name("nvbench_hash_join")
  .set_type_axes_names({"Key", "Value"})
  .set_max_noise(3)  // Custom noise: 3%. By default: 0.5%.
  .add_int64_axis("RSize", {50'000'000})
  .add_int64_axis("SSize", {50'000'000, 100'000'000, 200'000'000})
  .add_float64_axis("MatchingRate", {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0})
  .add_int64_axis("Multiplicity", {1, 8, 16, 32, 64})
  .add_float64_axis("Occupancy", {0.8});