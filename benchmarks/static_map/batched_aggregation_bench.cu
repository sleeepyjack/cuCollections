/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/detail/error.hpp>
#include <cuco/detail/utility/math.cuh>
#include <cuco/static_map.cuh>
#include <cuco/utility/key_generator.cuh>
#include <cuco/utility/reduction_functors.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef CUCO_ENABLE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

CUCO_DECLARE_BITWISE_COMPARABLE(nvbench::float64_t)


using namespace cuco::benchmark;  // defaults, dist_from_state
using namespace cuco::utility;    // key_generator, distribution

template <typename Key, typename Value>
void batched_aggregation(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  std::size_t const num_inputs  = state.get_int64("NumInputs");
  std::size_t const batch_size  = state.get_int64("BatchSize");
  std::size_t const cardinality = state.get_int64("Cardinality");
  std::size_t const num_streams = state.get_int64("NumStreams");

  if (num_inputs == 0 || batch_size == 0 || num_streams == 0 || cardinality == 0) {
    state.skip("NumInputs, BatchSize, NumStreams, and Cardinality must be greater than 0.");
    return;
  }
  if (cardinality > num_inputs) {
    state.skip("Cardinality must be <= NumInputs.");
    return;
  }

  // Derived average key multiplicity based on total inputs and desired cardinality.
  std::size_t const multiplicity = cuco::detail::int_div_ceil(num_inputs, cardinality);
  // Number of virtual batches processed across all streams.
  std::size_t const num_batches = cuco::detail::int_div_ceil(num_inputs, batch_size);

  // Pinned host storage: separate key and value arrays (SoA) to avoid pair alignment padding.
  Key* host_keys     = nullptr;
  Value* host_values = nullptr;
  CUCO_CUDA_TRY(cudaMallocHost(&host_keys, num_inputs * sizeof(Key)));
  CUCO_CUDA_TRY(cudaMallocHost(&host_values, num_inputs * sizeof(Value)));

  // Generate keys on the GPU into pinned host memory, fill values with 1.
  [[maybe_unused]] key_generator gen{};
  gen.generate<Key>(distribution::uniform{static_cast<int64_t>(multiplicity)},
                    thrust::device_pointer_cast(host_keys),
                    thrust::device_pointer_cast(host_keys) + num_inputs,
                    thrust::cuda::par);
  thrust::fill(thrust::cuda::par,
               thrust::device_pointer_cast(host_values),
               thrust::device_pointer_cast(host_values) + num_inputs,
               Value{1});

  state.add_element_count(num_inputs);
  // Report input bytes transferred: sizeof(Key) + sizeof(Value) per element.
  state.add_global_memory_reads<cuda::std::byte>(num_inputs * (sizeof(Key) + sizeof(Value)), "InputSize");

  Key constexpr empty_key_sentinel     = std::numeric_limits<Key>::max();
  Value constexpr empty_value_sentinel = Value{0};  // use the neutral element for the reduction
  // Target ~50% occupancy based on requested cardinality.
  std::size_t const map_capacity = cardinality / 0.5;

  cuco::static_map map{map_capacity,
                       cuco::empty_key<Key>{empty_key_sentinel},
                       cuco::empty_value<Value>{empty_value_sentinel}};

  // Create streams for strided batch processing.
  std::vector<cudaStream_t> streams(num_streams);
  for (auto& stream : streams) {
    CUCO_CUDA_TRY(cudaStreamCreate(&stream));
  }

  // Per-stream staging buffers for batch uploads (SoA: separate key and value vectors).
  std::vector<thrust::device_vector<Key>>   device_keys(num_streams, thrust::device_vector<Key>(batch_size));
  std::vector<thrust::device_vector<Value>> device_values(num_streams, thrust::device_vector<Value>(batch_size));

  // Timed region: batch uploads + insert_or_apply + retrieve_all.
  std::string rangeName = "aggregation: NumInputs=" + std::to_string(num_inputs) +
                          ", BatchSize=" + std::to_string(batch_size) +
                          ", Cardinality=" + std::to_string(cardinality) +
                          ", NumStreams=" + std::to_string(num_streams);
  PUSH_RANGE(rangeName.c_str(), 0)
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      // Strided assignment: each stream handles every num_streams-th batch.
      for (std::size_t stream_id = 0; stream_id < num_streams; ++stream_id) {
        auto& stream       = streams[stream_id];
        auto& batch_keys   = device_keys[stream_id];
        auto& batch_values = device_values[stream_id];
        auto batch_begin   = thrust::make_zip_iterator(
          thrust::make_tuple(batch_keys.begin(), batch_values.begin()));
          
        for (std::size_t batch = stream_id; batch < num_batches; batch += num_streams) {
          std::size_t const offset = batch * batch_size;
          std::size_t const count  = std::min(batch_size, num_inputs - offset);

          // H2D copy of keys and values separately into per-stream staging buffers.
          CUCO_CUDA_TRY(cudaMemcpyAsync(thrust::raw_pointer_cast(batch_keys.data()),
                                        host_keys + offset,
                                        count * sizeof(Key),
                                        cudaMemcpyHostToDevice,
                                        stream));
          CUCO_CUDA_TRY(cudaMemcpyAsync(thrust::raw_pointer_cast(batch_values.data()),
                                        host_values + offset,
                                        count * sizeof(Value),
                                        cudaMemcpyHostToDevice,
                                        stream));

          map.insert_or_apply_async(batch_begin,
                                    batch_begin + count,
                                    Value{0},
                                    cuco::reduce::plus{},
                                    cuda::stream_ref{stream});
        }
      }

      // Ensure all streams complete before retrieving results.
      for (auto& stream : streams) {
        CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
      }

      // Calculate the number of results to retrieve.
      auto const num_results = map.size({launch.get_stream()});

      // Retrieve results into device buffers.
      Key* result_keys     = nullptr;
      Value* result_values = nullptr;
      CUCO_CUDA_TRY(cudaMallocAsync(&result_keys, num_results * sizeof(Key), launch.get_stream()));
      CUCO_CUDA_TRY(
        cudaMallocAsync(&result_values, num_results * sizeof(Value), launch.get_stream()));
      map.retrieve_all(result_keys, result_values, {launch.get_stream()});
      timer.stop();

      POP_RANGE

      // Cleanup for next measurement.
      map.clear();
      CUCO_CUDA_TRY(cudaFreeAsync(result_keys, launch.get_stream()));
      CUCO_CUDA_TRY(cudaFreeAsync(result_values, launch.get_stream()));
    });

  for (auto& stream : streams) {
    CUCO_CUDA_TRY(cudaStreamDestroy(stream));
  }
  CUCO_CUDA_TRY(cudaFreeHost(host_keys));
  CUCO_CUDA_TRY(cudaFreeHost(host_values));
}

NVBENCH_BENCH_TYPES(batched_aggregation,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint8_t>,
                                      nvbench::type_list<nvbench::float64_t>))
  .set_name("static_map_batched_aggregation_uniform_uint8")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_axis("NumInputs", {1'000'000'000})
  .add_int64_axis("BatchSize", {100'000'000})
  .add_int64_axis("Cardinality", {1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, (1<<8) - 1})
  .add_int64_axis("NumStreams", {8});

NVBENCH_BENCH_TYPES(batched_aggregation,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint16_t>,
                                      nvbench::type_list<nvbench::float64_t>))
  .set_name("static_map_batched_aggregation_uniform_uint16")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_axis("NumInputs", {1'000'000'000})
  .add_int64_axis("BatchSize", {100'000'000})
  .add_int64_axis("Cardinality", {1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, (1<<16) - 1})
  .add_int64_axis("NumStreams", {8});

NVBENCH_BENCH_TYPES(batched_aggregation,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::uint32_t>,
                                      nvbench::type_list<nvbench::float64_t>))
  .set_name("static_map_batched_aggregation_uniform_uint32")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_axis("NumInputs", {1'000'000'000})
  .add_int64_axis("BatchSize", {100'000'000})
  .add_int64_axis("Cardinality", {1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24})
  .add_int64_axis("NumStreams", {8});
