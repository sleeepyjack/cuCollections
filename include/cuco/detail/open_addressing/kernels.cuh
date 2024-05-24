/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#pragma once

#include <cuco/detail/utility/cuda.cuh>
// #include <cuco/detail/bitwise_compare.cuh> TODOO

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include <cuda/functional>

#include <cooperative_groups.h>

#include <iterator>

namespace cuco::detail {
CUCO_SUPPRESS_KERNEL_WARNINGS

/**
 * @brief Inserts all elements in the range `[first, first + n)` and returns the number of
 * successful insertions if `pred` of the corresponding stencil returns true.
 *
 * @note If multiple elements in `[first, first + n)` compare equal, it is unspecified which element
 * is inserted.
 * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam AtomicT Atomic counter type
 * @tparam Ref Type of non-owning device container ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param num_successes Number of successful inserted elements
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename AtomicT,
          typename Ref>
CUCO_KERNEL void insert_if_n(InputIt first,
                             cuco::detail::index_type n,
                             StencilIt stencil,
                             Predicate pred,
                             AtomicT* num_successes,
                             Ref ref)
{
  using BlockReduce = cub::BlockReduce<typename Ref::size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  typename Ref::size_type thread_num_successes = 0;

  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      if constexpr (CGSize == 1) {
        if (ref.insert(insert_element)) { thread_num_successes++; };
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        if (ref.insert(tile, insert_element) && tile.thread_rank() == 0) { thread_num_successes++; }
      }
    }
    idx += loop_stride;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  auto const block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if (threadIdx.x == 0) {
    num_successes->fetch_add(block_num_successes, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Inserts all elements in the range `[first, first + n)` if `pred` of the corresponding
 * stencil returns true.
 *
 * @note If multiple elements in `[first, first + n)` compare equal, it is unspecified which element
 * is inserted.
 * @note The key `*(first + i)` is inserted if `pred( *(stencil + i) )` returns true.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename Ref>
CUCO_KERNEL void insert_if_n(
  InputIt first, cuco::detail::index_type n, StencilIt stencil, Predicate pred, Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    if (pred(*(stencil + idx))) {
      typename std::iterator_traits<InputIt>::value_type const& insert_element{*(first + idx)};
      if constexpr (CGSize == 1) {
        ref.insert(insert_element);
      } else {
        auto const tile =
          cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
        ref.insert(tile, insert_element);
      }
    }
    idx += loop_stride;
  }
}

/**
 * @brief Asynchronously erases keys in the range `[first, first + n)`.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize Number of threads in each block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param n Number of input elements
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize, int32_t BlockSize, typename InputIt, typename Ref>
CUCO_KERNEL void erase(InputIt first, cuco::detail::index_type n, Ref ref)
{
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  while (idx < n) {
    typename std::iterator_traits<InputIt>::value_type const& erase_element{*(first + idx)};
    if constexpr (CGSize == 1) {
      ref.erase(erase_element);
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<CGSize>(cooperative_groups::this_thread_block());
      ref.erase(tile, erase_element);
    }
    idx += loop_stride;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, first + n)` are contained in the data
 * structure if `pred` of the corresponding stencil returns true.
 *
 * @note If `pred( *(stencil + i) )` is true, stores `true` or `false` to `(output_begin + i)`
 * indicating if the key `*(first + i)` is present in the container. If `pred( *(stencil + i) )` is
 * false, stores false to `(output_begin + i)`.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam StencilIt Device accessible random access iterator whose value_type is
 * convertible to Predicate's argument type
 * @tparam Predicate Unary predicate callable whose return type must be convertible to `bool`
 * and argument type is convertible from `std::iterator_traits<StencilIt>::value_type`
 * @tparam OutputIt Device accessible output iterator assignable from `bool`
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys
 * @param stencil Beginning of the stencil sequence
 * @param pred Predicate to test on every element in the range `[stencil, stencil + n)`
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize,
          int32_t BlockSize,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename OutputIt,
          typename Ref>
CUCO_KERNEL void contains_if_n(InputIt first,
                               cuco::detail::index_type n,
                               StencilIt stencil,
                               Predicate pred,
                               OutputIt output_begin,
                               Ref ref)
{
  namespace cg = cooperative_groups;

  auto const block       = cg::this_thread_block();
  auto const thread_idx  = block.thread_rank();
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  __shared__ bool output_buffer[BlockSize / CGSize];

  while (idx - thread_idx < n) {  // the whole thread block falls into the same iteration
    if constexpr (CGSize == 1) {
      if (idx < n) {
        typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        output_buffer[thread_idx] = pred(*(stencil + idx)) ? ref.contains(key) : false;
      }
      block.sync();
      if (idx < n) { *(output_begin + idx) = output_buffer[thread_idx]; }
    } else {
      auto const tile = cg::tiled_partition<CGSize>(cg::this_thread_block());
      if (idx < n) {
        typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
        auto const found = pred(*(stencil + idx)) ? ref.contains(tile, key) : false;
        if (tile.thread_rank() == 0) { *(output_begin + idx) = found; }
      }
    }
    idx += loop_stride;
  }
}

/**
 * @brief Finds the equivalent container elements of all keys in the range `[first, first + n)`.
 *
 * @note If the key `*(first + i)` has a match in the container, copies the match to `(output_begin
 * + i)`. Else, copies the empty sentinel. Uses the CUDA Cooperative Groups API to leverage groups
 * of multiple threads to find each key. This provides a significant boost in throughput compared to
 * the non Cooperative Group `find` at moderate to high load factors.
 *
 * @tparam CGSize Number of threads in each CG
 * @tparam BlockSize The size of the thread block
 * @tparam InputIt Device accessible input iterator
 * @tparam OutputIt Device accessible output iterator
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of keys
 * @param n Number of keys to query
 * @param output_begin Beginning of the sequence of matched payloads retrieved for each key
 * @param ref Non-owning container device ref used to access the slot storage
 */
template <int32_t CGSize, int32_t BlockSize, typename InputIt, typename OutputIt, typename Ref>
CUCO_KERNEL void find(InputIt first, cuco::detail::index_type n, OutputIt output_begin, Ref ref)
{
  namespace cg = cooperative_groups;

  auto const block       = cg::this_thread_block();
  auto const thread_idx  = block.thread_rank();
  auto const loop_stride = cuco::detail::grid_stride() / CGSize;
  auto idx               = cuco::detail::global_thread_id() / CGSize;

  using output_type = typename std::iterator_traits<OutputIt>::value_type;
  __shared__ output_type output_buffer[BlockSize / CGSize];

  auto constexpr has_payload = not std::is_same_v<typename Ref::key_type, typename Ref::value_type>;

  auto const sentinel = [&]() {
    if constexpr (has_payload) {
      return ref.empty_value_sentinel();
    } else {
      return ref.empty_key_sentinel();
    }
  }();

  auto output = cuda::proclaim_return_type<output_type>([&] __device__(auto found) {
    if constexpr (has_payload) {
      return found == ref.end() ? sentinel : found->second;
    } else {
      return found == ref.end() ? sentinel : *found;
    }
  });

  while (idx - thread_idx < n) {  // the whole thread block falls into the same iteration
    if (idx < n) {
      typename std::iterator_traits<InputIt>::value_type const& key = *(first + idx);
      if constexpr (CGSize == 1) {
        auto const found = ref.find(key);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        output_buffer[thread_idx] = output(found);
        block.sync();
        *(output_begin + idx) = output_buffer[thread_idx];
      } else {
        auto const tile  = cg::tiled_partition<CGSize>(block);
        auto const found = ref.find(tile, key);

        if (tile.thread_rank() == 0) { *(output_begin + idx) = output(found); }
      }
    }
    idx += loop_stride;
  }
}

// TODOO enclose with scope
template <class CG, class Size, class ProbeOutputIt, class MatchOutputIt>
__device__ void flush_output_impl(
  CG const& group,
  Size buffer_size,
  cuda::std::iterator_traits<ProbeOutputIt>::value_type const* __restrict__ input_probe,
  cuda::std::iterator_traits<MatchOutputIt>::value_type const* __restrict__ input_match,
  ProbeOutputIt output_probe,
  MatchOutputIt output_match)
{
  auto i = group.thread_rank();

  // TODOO use memcpy_async when `thrust::is_contiguous_iterator_v<OutputIt> == true`
  while (i < buffer_size) {
    *(output_probe + i) = input_probe[i];
    *(output_match + i) = input_match[i];

    i += group.size();
  }
}

// TODOO
template <class CG,
          class Size,
          class ProbeOutputIt,
          class MatchOutputIt,
          cuda::thread_scope CounterScope>
__device__ void flush_output(
  CG const& tile,
  Size input_size,
  cuda::std::iterator_traits<ProbeOutputIt>::value_type const* __restrict__ input_probe,
  cuda::std::iterator_traits<MatchOutputIt>::value_type const* __restrict__ input_match,
  cuda::atomic<Size, CounterScope>& counter,
  ProbeOutputIt output_probe,
  MatchOutputIt output_match)
{
  Size offset;
#if defined(CUCO_HAS_CG_INVOKE_ONE)
  offset = cooperative_groups::invoke_one_broadcast(
    tile, [&]() { return counter.fetch_add(input_size, cuda::std::memory_order_relaxed); });
#else
  if (tile.thread_rank() == 0) {
    offset = counter.fetch_add(input_size, cuda::std::memory_order_relaxed);
  }
  offset = tile.shfl(offset, 0);
#endif
  flush_output_impl(
    tile, input_size, input_probe, input_match, output_probe + offset, output_match + offset);
}

template <typename Size,
          typename ProbeOutputIt,
          typename MatchOutputIt,
          cuda::thread_scope CounterScope>
__device__ void flush_output(
  cooperative_groups::thread_block const& block,
  Size input_size,
  cuda::std::iterator_traits<ProbeOutputIt>::value_type const* __restrict__ input_probe,
  cuda::std::iterator_traits<MatchOutputIt>::value_type const* __restrict__ input_match,
  cuda::atomic<Size, CounterScope>& counter,
  ProbeOutputIt output_probe,
  MatchOutputIt output_match)
{
  auto i = block.thread_rank();
  __shared__ Size offset;

#if defined(CUCO_HAS_CG_INVOKE_ONE)
  cooperative_groups::invoke_one(
    block, [&]() { offset = counter->fetch_add(input_size, cuda::std::memory_order_relaxed); });
#else
  if (i == 0) { offset = counter->fetch_add(input_size, cuda::std::memory_order_relaxed); }
#endif
  block.sync();
  flush_output_impl(
    block, input_size, input_probe, input_match, output_probe + offset, output_match + offset);
}

// TODOO
template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename CounterScope,
          typename Ref>
__device__ void group_retrieve(InputIt first,
                               cuco::detail::index_type n,
                               OutputIt1 output_probe,
                               OutputIt2 output_match,
                               cuda::atomic<Size, CounterScope>* counter,
                               Ref ref)
{
  namespace cg = cooperative_groups;

  using size_type = typename Ref::size_type;
  using ProbeKey  = typename std::iterator_traits<InputIt>::value_type;
  using Key       = typename Ref::key_type;

  auto constexpr tile_size   = Ref::cg_size;
  auto constexpr window_size = Ref::window_size;

  auto idx          = cuco::detail::global_thread_id() / tile_size;
  auto const stride = cuco::detail::grid_stride() / tile_size;
  auto const block  = cg::this_thread_block();
  auto const tile   = cg::tiled_partition<tile_size>(block);

  auto constexpr flushing_tile_size = cuco::detail::warp_size() / window_size;
  // random choice to tune
  auto constexpr flushing_buffer_size = 2 * flushing_tile_size;
  auto constexpr num_flushing_tiles   = BlockSize / flushing_tile_size;
  auto constexpr max_matches          = flushing_tile_size / tile_size;

  static_assert(flushing_tile_size > 0);

  auto const flushing_tile    = cg::tiled_partition<flushing_tile_size>(block);
  auto const flushing_tile_id = flushing_tile.meta_group_rank();

  __shared__ cuco::pair<ProbeKey, Key> flushing_tile_buffer[num_flushing_tiles][flushing_tile_size];

  using atomic_counter_type = cuda::atomic<size_type, cuda::thread_scope_block>;
  // per flushing-tile counter to track number of filled elements
  __shared__ atomic_counter_type flushing_counter[num_flushing_tiles];

#if defined(CUCO_HAS_CG_INVOKE_ONE)
  cg::invoke_one(flushing_tile,
                 [&]() { new (&flushing_counter[flushing_tile_id]) atomic_counter_type{0}; });
#else
  if (flushing_tile.thread_rank() == 0) {
    new (&flushing_counter[flushing_tile_id]) atomic_counter_type{0};
  }
#endif
  flushing_tile.sync();  // sync still needed since cg.any doesn't imply a memory barrier

  while (flushing_tile.any(idx < n)) {
    bool active_flag = idx < n;
    auto const active_flushing_tile =
      cg::binary_partition<flushing_tile_size>(flushing_tile, active_flag);
    if (active_flag) {
      auto const found = ref.find(tile, *(first + idx));
#if defined(CUCO_HAS_CG_INVOKE_ONE)
      if (found != ref.end()) {
        cg::invoke_one(tile, [&]() {
          auto const offset =
            flushing_counter[flushing_tile_id].fetch_add(1, cuda::std::memory_order_relaxed);
          flushing_tile_buffer[flushing_tile_id][offset] = {*(first + idx), *found};
        });
      }
#else
      if (tile.thread_rank() == 0 and found != ref.end()) {
        auto const offset =
          flushing_counter[flushing_tile_id].fetch_add(1, cuda::std::memory_order_relaxed);
        flushing_tile_buffer[flushing_tile_id][offset] = {*(first + idx), *found};
      }
#endif
    }

    flushing_tile.sync();
    auto const buffer_size =
      flushing_counter[flushing_tile_id].load(cuda::std::memory_order_relaxed);
    if (buffer_size + max_matches > flushing_buffer_size) {
      flush_buffer(flushing_tile,
                   buffer_size,
                   flushing_tile_buffer[flushing_tile_id],
                   counter,
                   output_probe,
                   output_match);
      flushing_tile.sync();
#if defined(CUCO_HAS_CG_INVOKE_ONE)
      cg::invoke_one(flushing_tile, [&]() {
        flushing_counter[flushing_tile_id].store(0, cuda::std::memory_order_relaxed);
      });
#else
      if (flushing_tile.thread_rank() == 0) {
        flushing_counter[flushing_tile_id].store(0, cuda::std::memory_order_relaxed);
      }
#endif
      flushing_tile.sync();
    }

    idx += stride;
  }  // while

  auto const buffer_size = flushing_counter[flushing_tile_id].load(cuda::std::memory_order_relaxed);
  if (buffer_size > 0) {
    flush_buffer(flushing_tile,
                 buffer_size,
                 flushing_tile_buffer[flushing_tile_id],
                 counter,
                 output_probe,
                 output_match);
  }
}

// TODOO
template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename AtomicT,
          typename Ref>
__device__ void scalar_retrieve(InputIt first,
                                cuco::detail::index_type n,
                                OutputIt1 output_probe,
                                OutputIt2 output_match,
                                AtomicT* counter,
                                Ref ref)
{
  namespace cg = cooperative_groups;

  using size_type = typename Ref::size_type;
  using ProbeKey  = typename std::iterator_traits<InputIt>::value_type;
  using Key       = typename Ref::key_type;

  auto idx          = cuco::detail::global_thread_id();
  auto const stride = cuco::detail::grid_stride();
  auto const block  = cg::this_thread_block();

  using block_scan = cub::BlockScan<size_type, BlockSize>;
  __shared__ typename block_scan::TempStorage block_scan_temp_storage;

  auto constexpr buffer_capacity = 2 * BlockSize;  // TODO
  __shared__ std::iterator_traits<OutputIt1>::value_type probe_buffer[buffer_capacity];
  __shared__ std::iterator_traits<OutputIt2>::value_type match_buffer[buffer_capacity];

  __shared__ cuco::pair<ProbeKey, Key> buffer[buffer_capacity];
  size_type buffer_size = 0;

  while (idx - block.thread_rank() < n) {  // the whole thread block falls into the same iteration
    auto const found     = idx < n ? ref.find(*(first + idx)) : ref.end();
    auto const has_match = found != ref.end();

    // Use a whole-block scan to calculate the output location
    size_type offset;
    size_type block_count;
    block_scan(block_scan_temp_storage)
      .ExclusiveSum(static_cast<size_type>(has_match), offset, block_count);

    if (buffer_size + block_count > buffer_capacity) {
      flush_buffer(block, buffer_size, buffer, counter, output_probe, output_match);
      block.sync();
      buffer_size = 0;
    }

    if (has_match) { buffer[buffer_size + offset] = {*(first + idx), *found}; }
    buffer_size += block_count;
    block.sync();

    idx += stride;
  }  // while

  if (buffer_size > 0) {
    flush_buffer(block, buffer_size, buffer, counter, output_probe, output_match);
  }
}

// TODOO
template <int32_t BlockSize,
          typename InputIt,
          typename OutputIt1,
          typename OutputIt2,
          typename AtomicT,
          typename Ref>
CUCO_KERNEL void retrieve(InputIt first,
                          cuco::detail::index_type n,
                          OutputIt1 output_probe,
                          OutputIt2 output_match,
                          AtomicT* counter,
                          Ref ref)
{
  // Scalar retrieve without using CG
  if constexpr (Ref::cg_size == 1) {
    scalar_retrieve<BlockSize>(first, n, output_probe, output_match, counter, ref);
  } else {
    group_retrieve<BlockSize>(first, n, output_probe, output_match, counter, ref);
  }
}

/**
 * @brief Calculates the number of filled slots for the given window storage.
 *
 * @tparam BlockSize Number of threads in each block
 * @tparam StorageRef Type of non-owning ref allowing access to storage
 * @tparam Predicate Type of predicate indicating if the given slot is filled
 * @tparam AtomicT Atomic counter type
 *
 * @param storage Non-owning device ref used to access the slot storage
 * @param is_filled Predicate indicating if the given slot is filled
 * @param count Number of filled slots
 */
template <int32_t BlockSize, typename StorageRef, typename Predicate, typename AtomicT>
CUCO_KERNEL void size(StorageRef storage, Predicate is_filled, AtomicT* count)
{
  using size_type = typename StorageRef::size_type;

  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  size_type thread_count = 0;
  auto const n           = storage.num_windows();

  while (idx < n) {
    auto const window = storage[idx];
#pragma unroll
    for (auto const& it : window) {
      thread_count += static_cast<size_type>(is_filled(it));
    }
    idx += loop_stride;
  }

  using BlockReduce = cub::BlockReduce<size_type, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto const block_count = BlockReduce(temp_storage).Sum(thread_count);
  if (threadIdx.x == 0) { count->fetch_add(block_count, cuda::std::memory_order_relaxed); }
}

template <int32_t BlockSize, typename ContainerRef, typename Predicate>
CUCO_KERNEL void rehash(typename ContainerRef::storage_ref_type storage_ref,
                        ContainerRef container_ref,
                        Predicate is_filled)
{
  namespace cg = cooperative_groups;

  __shared__ typename ContainerRef::value_type buffer[BlockSize];
  __shared__ unsigned int buffer_size;

  auto constexpr cg_size = ContainerRef::cg_size;
  auto const block       = cg::this_thread_block();
  auto const tile        = cg::tiled_partition<cg_size>(block);

  auto const thread_rank         = block.thread_rank();
  auto constexpr tiles_per_block = BlockSize / cg_size;  // tile.meta_group_size() but constexpr
  auto const tile_rank           = tile.meta_group_rank();
  auto const loop_stride         = cuco::detail::grid_stride();
  auto idx                       = cuco::detail::global_thread_id();
  auto const n                   = storage_ref.num_windows();

  while (idx - thread_rank < n) {
    if (thread_rank == 0) { buffer_size = 0; }
    block.sync();

    // gather values in shmem buffer
    if (idx < n) {
      auto const window = storage_ref[idx];

      for (auto const& slot : window) {
        if (is_filled(slot)) { buffer[atomicAdd_block(&buffer_size, 1)] = slot; }
      }
    }
    block.sync();

    auto const local_buffer_size = buffer_size;

    // insert from shmem buffer into the container
    for (auto tidx = tile_rank; tidx < local_buffer_size; tidx += tiles_per_block) {
      container_ref.insert(tile, buffer[tidx]);
    }
    block.sync();

    idx += loop_stride;
  }
}

}  // namespace cuco::detail
