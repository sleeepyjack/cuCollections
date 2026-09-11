/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/bucket_storage.cuh>
#include <cuco/probing_scheme.cuh>

#include <cuda/std/algorithm>
#include <cuda/std/bit>

#include <cassert>
#include <cstddef>

namespace cuco::detail {

/**
 * @brief Identifies the built-in flat bucket storage.
 * @tparam Storage Storage reference type
 */
template <typename Storage>
inline constexpr bool is_bucket_storage_ref_v = false;

template <typename T, int B, typename Extent>
/// Native bucket storage uses a fixed byte stride.
inline constexpr bool is_bucket_storage_ref_v<bucket_storage_ref<T, B, Extent>> = true;

/**
 * @brief Identifies probing schemes whose iterators preserve bucket boundaries.
 * @tparam Probe Probing scheme type
 */
template <typename Probe>
inline constexpr bool is_bucket_aligned_probing_v = false;

template <int CG, typename Hash>
/// Linear probing preserves bucket boundaries.
inline constexpr bool is_bucket_aligned_probing_v<linear_probing<CG, Hash>> = true;

template <int CG, typename Hash1, typename Hash2>
/// Double hashing preserves bucket boundaries.
inline constexpr bool is_bucket_aligned_probing_v<double_hashing<CG, Hash1, Hash2>> = true;

/**
 * @brief Loads a bucket using the alignment guaranteed by native probing.
 *
 * Custom storage and probing schemes retain their ordinary slot-indexed access.
 * The native schemes initialize, advance, and wrap in multiples of the bucket
 * size, so no runtime alignment branch is necessary.
 *
 * @tparam MaxLoadBytes Maximum alignment to expose for this access
 * @tparam Storage Storage reference type
 * @tparam Probe Probing scheme type
 * @param storage Slot storage
 * @param index Slot index produced by the probing iterator
 * @return The bucket at `index`
 */
template <std::size_t MaxLoadBytes = 32, typename Storage, typename Probe>
[[nodiscard]] __device__ constexpr typename Storage::bucket_type load_bucket(
  Storage const& storage, typename Storage::size_type index, Probe const&) noexcept
{
  static_assert(cuda::std::has_single_bit(MaxLoadBytes), "Load alignment must be a power of two");
  if constexpr (is_bucket_storage_ref_v<Storage> && is_bucket_aligned_probing_v<Probe>) {
    assert(index % Storage::bucket_size == 0);
    assert(index <= storage.capacity() && Storage::bucket_size <= storage.capacity() - index);
    constexpr auto alignment = cuda::std::min(Storage::alignment, MaxLoadBytes);
    if constexpr (alignment <= alignof(typename Storage::value_type)) {
      return storage[index];
    } else {
      auto const* ptr = __builtin_assume_aligned(storage.data() + index, alignment);
      return *static_cast<typename Storage::bucket_type const*>(ptr);
    }
  } else {
    return storage[index];
  }
}

}  // namespace cuco::detail
