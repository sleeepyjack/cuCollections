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

using namespace cuco::benchmark;  // defaults, dist_from_state
using namespace cuco::utility;    // key_generator, distribution

template <typename Key, typename Value>
void batched_aggregation(nvbench::state& state, nvbench::type_list<Key, Value>)
{
  using pair_type = cuco::pair<Key, Value>;

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

  // Pinned host storage for input pairs.
  pair_type* host_pairs = nullptr;
  CUCO_CUDA_TRY(cudaMallocHost(&host_pairs, num_inputs * sizeof(pair_type)));

  [[maybe_unused]] key_generator gen{};
  auto device_pairs = thrust::device_pointer_cast(host_pairs);
  auto out_iter     = thrust::make_transform_output_iterator(
    device_pairs, [] __host__ __device__(Key key) { return pair_type{key, Value{1}}; });
  // Generate keys on the GPU and materialize (key, 1) pairs in pinned host memory.
  gen.generate<Key>(distribution::uniform{static_cast<int64_t>(multiplicity)},
                    out_iter,
                    out_iter + num_inputs,
                    thrust::cuda::par);

  state.add_element_count(num_inputs);
  state.add_global_memory_reads<pair_type>(num_inputs, "InputSize");

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

  // Per-stream staging buffers for batch uploads.
  std::vector<thrust::device_vector<pair_type>> device_batches;
  device_batches.reserve(num_streams);
  for (std::size_t i = 0; i < num_streams; ++i) {
    device_batches.emplace_back(batch_size);
  }

  // Timed region: batch uploads + insert_or_apply + retrieve_all.
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      // Strided assignment: each stream handles every num_streams-th batch.
      for (std::size_t stream_id = 0; stream_id < num_streams; ++stream_id) {
        auto& stream      = streams[stream_id];
        auto& batch_pairs = device_batches[stream_id];
        for (std::size_t batch = stream_id; batch < num_batches; batch += num_streams) {
          std::size_t const offset = batch * batch_size;
          std::size_t const count  = std::min(batch_size, num_inputs - offset);

          // H2D copy of the next batch into the per-stream staging buffer.
          CUCO_CUDA_TRY(cudaMemcpyAsync(thrust::raw_pointer_cast(batch_pairs.data()),
                                        host_pairs + offset,
                                        count * sizeof(pair_type),
                                        cudaMemcpyHostToDevice,
                                        stream));

          // Insert-or-apply for this batch on the stream.
          map.insert_or_apply_async(batch_pairs.begin(),
                                    batch_pairs.begin() + count,
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

      // Cleanup for next measurement.
      map.clear();
      CUCO_CUDA_TRY(cudaFreeAsync(result_keys, launch.get_stream()));
      CUCO_CUDA_TRY(cudaFreeAsync(result_values, launch.get_stream()));
    });

  for (auto& stream : streams) {
    CUCO_CUDA_TRY(cudaStreamDestroy(stream));
  }
  CUCO_CUDA_TRY(cudaFreeHost(host_pairs));
}

NVBENCH_BENCH_TYPES(batched_aggregation,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>))
  .set_name("static_map_batched_aggregation_uniform")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_axis("NumInputs", {1'000'000'000})
  .add_int64_axis("BatchSize", {25'000'000, 50'000'000})
  .add_int64_axis("Cardinality", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
  .add_int64_axis("NumStreams", {8});
