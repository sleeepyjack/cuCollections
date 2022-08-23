/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <benchmark/benchmark.h>

#include <key_generator.hpp>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#include <fstream>  // TODO remove

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void generate_size_and_occupancy(benchmark::internal::Benchmark* b)  // TODO remove
{
  for (auto size = 100'000'000; size <= 100'000'000; size *= 10) {
    for (auto occupancy = 10; occupancy <= 90; occupancy += 10) {
      b->Args({size, occupancy});
    }
  }
}

template <typename Key, typename Value, dist_type Dist>
static void BM_static_map_insert(::benchmark::State& state)
{
  using map_type = cuco::static_map<Key, Value>;

  auto const num_keys   = state.range(0);
  float const occupancy = state.range(1) / float{100};  // TODO auto?

  key_generator<Dist> gen;

  std::size_t const size = num_keys / occupancy;

  thrust::device_vector<Key> keys(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> pairs(num_keys);

  for (auto _ : state) {
    gen.generate(thrust::device, keys.begin(), keys.end());

    thrust::transform(thrust::device,
                      keys.cbegin(),
                      keys.cend(),
                      pairs.begin(),
                      [] __host__ __device__(Key const& key) {
                        return cuco::make_pair<Key, Value>(key, Value(42));
                      });

    map_type map{size, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    map.insert(pairs.cbegin(), pairs.cend());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    state.SetIterationTime(ms / 1000);
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_static_map_search_all(::benchmark::State& state)
{
  using map_type = cuco::static_map<Key, Value>;

  std::size_t num_keys = state.range(0);
  float occupancy      = state.range(1) / float{100};
  std::size_t size     = num_keys / occupancy;

  map_type map{size, cuco::sentinel::empty_key<Key>{-1}, cuco::sentinel::empty_value<Value>{-1}};

  std::vector<Key> h_keys(num_keys);
  std::vector<Value> h_values(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);
  std::vector<Value> h_results(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (std::size_t i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<Value> d_results(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  map.insert(d_pairs.begin(), d_pairs.end());

  for (auto _ : state) {
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
    // TODO: get rid of sync and rewrite the benchmark with `nvbench`
    // once https://github.com/NVIDIA/nvbench/pull/80 is merged
    cudaDeviceSynchronize();
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

template <typename Key, typename Value, dist_type Dist>
static void BM_static_map_erase_all(::benchmark::State& state)
{
  using map_type = cuco::static_map<Key, Value>;

  std::size_t num_keys = state.range(0);
  float occupancy      = state.range(1) / float{100};
  std::size_t size     = num_keys / occupancy;

  // static map with erase support
  map_type map{size,
               cuco::sentinel::empty_key<Key>{-1},
               cuco::sentinel::empty_value<Value>{-1},
               cuco::sentinel::erased_key<Key>{-2}};

  std::vector<Key> h_keys(num_keys);
  std::vector<Value> h_values(num_keys);
  std::vector<cuco::pair_type<Key, Value>> h_pairs(num_keys);
  std::vector<Value> h_results(num_keys);

  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());

  for (std::size_t i = 0; i < num_keys; ++i) {
    Key key           = h_keys[i];
    Value val         = h_keys[i];
    h_pairs[i].first  = key;
    h_pairs[i].second = val;
  }

  thrust::device_vector<Key> d_keys(h_keys);
  thrust::device_vector<bool> d_results(num_keys);
  thrust::device_vector<cuco::pair_type<Key, Value>> d_pairs(h_pairs);

  for (auto _ : state) {
    state.PauseTiming();
    map.insert(d_pairs.begin(), d_pairs.end());
    state.ResumeTiming();

    map.erase(d_keys.begin(), d_keys.end());
  }

  state.SetBytesProcessed((sizeof(Key) + sizeof(Value)) * int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int32_t, int32_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int32_t, int32_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int64_t, int64_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int64_t, int64_t, dist_type::UNIFORM)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_insert, int64_t, int64_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy)
  ->UseManualTime();

BENCHMARK_TEMPLATE(BM_static_map_search_all, int64_t, int64_t, dist_type::GAUSSIAN)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);

BENCHMARK_TEMPLATE(BM_static_map_erase_all, int32_t, int32_t, dist_type::UNIQUE)
  ->Unit(benchmark::kMillisecond)
  ->Apply(generate_size_and_occupancy);
