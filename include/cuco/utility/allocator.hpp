/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cuco/detail/error.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

#include <unordered_map>

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cudaCompressibleMemory/compMalloc.cpp
namespace {
cudaError_t setProp(CUmemAllocationProp* prop, bool UseCompressibleMemory)
{
  cudaFree(0); // init context
  CUdevice currentDevice;
  if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS) return cudaErrorMemoryAllocation;

  memset(prop, 0, sizeof(CUmemAllocationProp));
  prop->type          = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id   = currentDevice;

  if (UseCompressibleMemory) prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

  return cudaSuccess;
}

cudaError_t allocateCompressible(void** adr, size_t size, bool UseCompressibleMemory)
{
  CUmemAllocationProp prop = {};
  cudaError_t err          = setProp(&prop, UseCompressibleMemory);
  if (err != cudaSuccess) return err;

  size_t granularity = 0;
  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
      CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;
  size = ((size - 1) / granularity + 1) * granularity;
  CUdeviceptr dptr;
  if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS) return cudaErrorMemoryAllocation;

  CUmemGenericAllocationHandle allocationHandle;
  if (cuMemCreate(&allocationHandle, size, &prop, 0) != CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;

  // Check if cuMemCreate was able to allocate compressible memory.
  if (UseCompressibleMemory) {
    CUmemAllocationProp allocationProp = {};
    cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
    if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
      printf("Could not allocate compressible memory... so waiving execution\n");
      exit(2);
    }
  }

  if (cuMemMap(dptr, size, 0, allocationHandle, 0) != CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;

  if (cuMemRelease(allocationHandle) != CUDA_SUCCESS) return cudaErrorMemoryAllocation;

  CUmemAccessDesc accessDescriptor;
  accessDescriptor.location.id   = prop.location.id;
  accessDescriptor.location.type = prop.location.type;
  accessDescriptor.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;

  *adr = (void*)dptr;
  return cudaSuccess;
}

cudaError_t freeCompressible(void* ptr, size_t size, bool UseCompressibleMemory)
{
  CUmemAllocationProp prop = {};
  cudaError_t err          = setProp(&prop, UseCompressibleMemory);
  if (err != cudaSuccess) return err;

  size_t granularity = 0;
  if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
      CUDA_SUCCESS)
    return cudaErrorMemoryAllocation;
  size = ((size - 1) / granularity + 1) * granularity;

  if (ptr == NULL) return cudaSuccess;
  if (cuMemUnmap((CUdeviceptr)ptr, size) != CUDA_SUCCESS ||
      cuMemAddressFree((CUdeviceptr)ptr, size) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}
}  // namespace

namespace cuco {
/**
 * @brief A device allocator using `cudaMalloc`/`cudaFree` to satisfy (de)allocations.
 *
 * @tparam T The allocator's value type
 */
template <typename T>
class cuda_allocator {
 public:
  using value_type = T;  ///< Allocator's value type

  cuda_allocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  {
  }

  /**
   * @brief Allocates storage for `n` objects of type `T` using `cudaMalloc`.
   *
   * @param n The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n)
  {
    value_type* p;
    CUCO_CUDA_TRY(allocateCompressible(reinterpret_cast<void**>(&p), sizeof(value_type) * n, true));
    get_allocations()[reinterpret_cast<void*>(p)] = sizeof(value_type) * n;
    return p;
  }

  /**
   * @brief Deallocates storage pointed to by `p`.
   *
   * @param p Pointer to memory to deallocate
   */
  void deallocate(value_type* p, std::size_t)
  {
    CUCO_CUDA_TRY(freeCompressible(reinterpret_cast<void*>(p), get_allocations()[reinterpret_cast<void*>(p)], true));
    get_allocations().erase(reinterpret_cast<void*>(p));
  }
 private:
  static std::unordered_map<void*, size_t>& get_allocations() {
    static std::unordered_map<void*, size_t> allocations;
    return allocations;
  }
};

template <typename T, typename U>
bool operator==(cuda_allocator<T> const&, cuda_allocator<U> const&) noexcept
{
  return true;
}

template <typename T, typename U>
bool operator!=(cuda_allocator<T> const& lhs, cuda_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace cuco
