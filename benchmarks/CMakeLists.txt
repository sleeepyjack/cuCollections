#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG main
  GIT_SHALLOW TRUE
)

###################################################################################################
### compiler function #############################################################################
###################################################################################################

###################################################################################################
function(ConfigureNVBench BENCH_NAME)
    add_executable(${BENCH_NAME} ${ARGN})
    set_target_properties(${BENCH_NAME} PROPERTIES
                                        POSITION_INDEPENDENT_CODE ON
                                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
    target_include_directories(${BENCH_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${BENCH_NAME} PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
    target_link_libraries(${BENCH_NAME} PRIVATE
                                        nvbench::main
                                        pthread
                                        cuco)
endfunction(ConfigureNVBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

###################################################################################################
# - dynamic_map benchmarks ------------------------------------------------------------------------
# set(DYNAMIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/dynamic_map_bench.cu")
# ConfigureBench(DYNAMIC_MAP_BENCH "${DYNAMIC_MAP_BENCH_SRC}")

###################################################################################################
# - static_map benchmarks -------------------------------------------------------------------------
# set(STATIC_MAP_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/hash_table/static_map_bench.cu")
# ConfigureBench(STATIC_MAP_BENCH "${STATIC_MAP_BENCH_SRC}")

###################################################################################################
# - static_multimap benchmarks --------------------------------------------------------------------
# ConfigureNVBench(STATIC_MULTIMAP_BENCH
  # hash_table/static_multimap/count_bench.cu)
  # hash_table/static_multimap/insert_bench.cu
  # hash_table/static_multimap/pair_retrieve_bench.cu
  # hash_table/static_multimap/query_bench.cu
  # hash_table/static_multimap/retrieve_bench.cu)

# ConfigureNVBench(RETRIEVE_BENCH
#   hash_table/static_multimap/optimal_retrieve_bench.cu)

###################################################################################################
# - reduce_by_key benchmarks ----------------------------------------------------------------------
# set(RBK_BENCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_by_key/reduce_by_key.cu")
# ConfigureBench(RBK_BENCH "${RBK_BENCH_SRC}")

ConfigureNVBench(STATIC_MAP_BENCH
  hash_table/static_map/insert_bench.cu
  hash_table/static_map/find_bench.cu
  hash_table/static_map/erase_bench.cu
  hash_table/static_map/retrieve_all_bench.cu)