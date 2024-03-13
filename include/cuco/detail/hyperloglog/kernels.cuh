/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cuco/utility/cuda_thread_scope.cuh>

#include <cuda/std/span>

#include <cstddef>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda/pipeline>

namespace cuco::hyperloglog_ns::detail {
CUCO_SUPPRESS_KERNEL_WARNINGS

template <class RefType>
CUCO_KERNEL void clear(RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) { ref.clear(block); }
}

template <class RefType>
CUCO_KERNEL void add_shmem_pipelined(typename RefType::value_type* __restrict__ const first,
                                     cuco::detail::index_type n,
                                     RefType ref)
{
  using value_type     = typename RefType::value_type;
  using local_ref_type = typename RefType::with_scope<cuda::thread_scope_block>;

  auto const grid   = cooperative_groups::this_grid();
  auto const block  = cooperative_groups::this_thread_block();
  auto const thread = cooperative_groups::this_thread();

  // TODO assert alignment
  extern __shared__ std::byte local_sketch[];

  local_ref_type local_ref(cuda::std::span{local_sketch, ref.sketch_bytes()}, {});
  local_ref.clear(block);
  block.sync();

  constexpr auto stages_count = 4;
  auto const batch_sz         = n / grid.size();

  __align__(16) __shared__ value_type shared[768 * stages_count];

  // No pipeline::shared_state needed
  cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

  auto block_batch = [&](size_t batch) -> int {
    return block.group_index().x * block.size() + grid.size() * batch;
  };

  for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
    for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
      pipeline.producer_acquire();
      size_t shared_idx = fetch_batch % stages_count;
      size_t batch_idx  = fetch_batch;
      // Each thread fetches its own data:
      size_t thread_batch_idx = block_batch(batch_idx) + threadIdx.x;
      // The copy is performed by a single `thread` and the size of the batch is now that of a
      // single element:
      cuda::memcpy_async(thread,
                         shared + shared_idx * block.size() + threadIdx.x,
                         first + thread_batch_idx,
                         sizeof(value_type),
                         pipeline);
      pipeline.producer_commit();
    }
    pipeline.consumer_wait();
    __syncwarp();
    // block.sync();  // __syncthreads: All memcpy_async of all threads in the block for this stage
    // have completed here
    int shared_idx = compute_batch % stages_count;
    local_ref.add(*(shared + shared_idx * block.size() + block.thread_rank()));
    pipeline.consumer_release();
  }
  // block.sync();

  // handle tail items
  int const tail_items = n % grid.size();
  auto const tail      = first + n - tail_items;
  if (grid.thread_rank() < tail_items) { local_ref.add(*(tail + grid.thread_rank())); }
  block.sync();

  ref.merge(block, local_ref);
}

template <class InputIt, class RefType>
CUCO_KERNEL void add_shmem(InputIt first, cuco::detail::index_type n, RefType ref)
{
  using local_ref_type = typename RefType::with_scope<cuda::thread_scope_block>;

  // TODO assert alignment
  extern __shared__ std::byte local_sketch[];

  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();
  auto const block       = cooperative_groups::this_thread_block();

  local_ref_type local_ref(cuda::std::span{local_sketch, ref.sketch_bytes()}, {});
  local_ref.clear(block);
  block.sync();

  while (idx < n) {
    local_ref.add(*(first + idx));
    idx += loop_stride;
  }
  block.sync();

  ref.merge(block, local_ref);
}

template <class InputIt, class RefType>
CUCO_KERNEL void add_gmem(InputIt first, cuco::detail::index_type n, RefType ref)
{
  auto const loop_stride = cuco::detail::grid_stride();
  auto idx               = cuco::detail::global_thread_id();

  while (idx < n) {
    ref.add(*(first + idx));
    idx += loop_stride;
  }
}

template <class OtherRefType, class RefType>
CUCO_KERNEL void merge(OtherRefType other_ref, RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) { ref.merge(block, other_ref); }
}

// TODO this kernel currently isn't being used
template <class RefType>
CUCO_KERNEL void estimate(std::size_t* cardinality, RefType ref)
{
  auto const block = cooperative_groups::this_thread_block();
  if (block.group_index().x == 0) {
    auto const estimate = ref.estimate(block);
    if (block.thread_rank() == 0) { *cardinality = estimate; }
  }
}
}  // namespace cuco::hyperloglog_ns::detail