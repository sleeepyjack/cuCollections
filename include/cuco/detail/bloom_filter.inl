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

namespace cuco {

template <typename Key, cuda::thread_scope Scope, typename Allocator>
bloom_filter<Key, Scope, Allocator>::bloom_filter()
  : slot_deleter_{slot_allocator_, num_slots_, stream_, cache_hit_ratio_},
    slots_{nullptr, slot_deleter_}
{
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
bloom_filter<Key, Scope, Allocator>::bloom_filter(std::size_t num_bits,
                                                  std::size_t num_hashes,
                                                  cudaStream_t stream,
                                                  Allocator const& alloc,
                                                  float cache_hit_ratio)
  : num_bits_{SDIV(num_bits, detail::type_bits<slot_type>()) * detail::type_bits<slot_type>()},
    num_slots_{SDIV(num_bits, detail::type_bits<slot_type>())},
    num_hashes_{std::clamp(num_hashes, std::size_t{1}, detail::type_bits<slot_type>())},
    stream_{stream},
    cache_hit_ratio_{cache_hit_ratio},
    slot_allocator_{alloc},
    slot_deleter_{slot_allocator_, num_slots_, stream_, cache_hit_ratio_},
    slots_{slot_allocator_.allocate(num_slots_, stream_), slot_deleter_}
{
  initialize(stream_);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
void bloom_filter<Key, Scope, Allocator>::initialize(cudaStream_t stream)
{
  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_slots_, stride * block_size);

  detail::initialize<block_size>
    <<<grid_size, block_size, 0, stream>>>(get_device_mutable_view(), cache_hit_ratio_);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename Hash>
void bloom_filter<Key, Scope, Allocator>::insert(InputIt first,
                                                 InputIt last,
                                                 cudaStream_t stream,
                                                 Hash hash)
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0 or get_num_slots() == 0) { return; }

  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_keys, stride * block_size);
  detail::insert<block_size>
    <<<grid_size, block_size, 0, stream>>>(first, last, get_device_mutable_view(), hash);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
template <typename InputIt, typename OutputIt, typename Hash>
void bloom_filter<Key, Scope, Allocator>::contains(
  InputIt first, InputIt last, OutputIt output_begin, cudaStream_t stream, Hash hash)
{
  auto num_keys = std::distance(first, last);
  if (num_keys == 0 or get_num_slots() == 0) { return; }

  std::size_t constexpr block_size = 256;
  std::size_t constexpr stride     = 4;
  std::size_t const grid_size      = SDIV(num_keys, stride * block_size);
  detail::contains<block_size>
    <<<grid_size, block_size, 0, stream>>>(first, last, output_begin, get_device_view(), hash);
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
template <typename Hash>
__device__ auto bloom_filter<Key, Scope, Allocator>::device_view_base::key_pattern(
  Key const& key, Hash hash) const noexcept
{
  // TODO find a better solution for those secondary hashers
  cuco::detail::MurmurHash3_32<Key> hash2{42};
  cuco::detail::MurmurHash3_32<std::size_t> hash3{43};
  slot_type pattern{0};
  std::size_t k{0};
  std::size_t i{0};
  auto const h  = hash(key);
  auto const h2 = hash2(key);  // seed mitigates secondary clustering

  while (k < num_hashes_) {
    slot_type const bit = slot_type{1} << ((h + hash3(h2 + i)) % detail::type_bits<slot_type>());

    if (not(pattern & bit)) {
      pattern += bit;
      k++;
    }
    i++;
  }
  return pattern;
}

template <typename Key, cuda::thread_scope Scope, typename Allocator>
template <typename Hash>
__device__ bool bloom_filter<Key, Scope, Allocator>::device_mutable_view::insert(Key const& key,
                                                                                 Hash hash) noexcept
{
  if (this->get_num_slots() == 0) { return false; }

  iterator const slot     = key_slot(key, hash);
  slot_type const pattern = key_pattern(key, hash);
  slot_type result;

#if defined(CUCO_HAS_L2_RESIDENCY_CONTROL)
  float const hit_ratio = this->get_cache_hit_ratio();
  if (__isGlobal(slot) and hit_ratio > 0.0) {
    std::uint64_t descriptor;
    // clang-format off
    asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, %1;" : "=l"(descriptor) : "f"(hit_ratio));
    asm volatile("atom.or.relaxed.L2::cache_hint.gpu.b64 %0, [%1], %2, %3;" : "=l"(result) : "l"(slot), "l"(pattern), "l"(descriptor));
    // clang-format on
  } else {
    result = slot->fetch_or(key_pattern(key, hash), cuda::memory_order_relaxed);
  }
#else
  result = slot->fetch_or(key_pattern(key, hash), cuda::memory_order_relaxed);
#endif

  // return `true` if the key's pattern was not already present in the filter,
  // else return `false`.
  return (result & pattern) != pattern;
}  // namespace cuco

template <typename Key, cuda::thread_scope Scope, typename Allocator>
template <typename Hash>
__device__ bool bloom_filter<Key, Scope, Allocator>::device_view::contains(Key const& key,
                                                                           Hash hash) const noexcept
{
  if (this->get_num_slots() == 0) { return false; }

  const_iterator const slot = key_slot(key, hash);
  slot_type const pattern   = key_pattern(key, hash);
  slot_type result;

#if defined(CUCO_HAS_L2_RESIDENCY_CONTROL)
  float const hit_ratio = this->get_cache_hit_ratio();
  if (__isGlobal(slot) and hit_ratio > 0.0) {
    std::uint64_t descriptor;
    // clang-format off
    asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, %1;" : "=l"(descriptor) : "f"(hit_ratio));
    asm volatile("ld.relaxed.L2::cache_hint.gpu.b64 %0, [%1], %2;" : "=l"(result) : "l"(slot), "l"(descriptor));
    // clang-format on
  } else {
    result = slot->load(cuda::memory_order_relaxed);
  }
#else
  result = slot->load(cuda::memory_order_relaxed);
#endif

  // return `true` if the key's pattern was already present in the filter,
  // else return `false`.
  return (result & pattern) == pattern;
}
}  // namespace cuco
