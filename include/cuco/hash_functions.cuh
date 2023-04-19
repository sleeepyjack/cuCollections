/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuco/detail/probe_sequence_impl.cuh>

#include <type_traits>

namespace cuco {

/**
 * @brief A `murmurhash3_32` hash function to hash the given argument on host and device.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using murmurhash3_32 = detail::MurmurHash3_32<Key>;

// TODO docs
template <typename Key>
using fmix_32 = detail::MurmurHash3_fmix32<Key>;

template <typename Key>
using fmix_64 = detail::MurmurHash3_fmix64<Key>;

template <typename Key>
using default_hash_function =
  std::conditional_t<sizeof(Key) == 4,
                     fmix_32<Key>,
                     std::conditional_t<sizeof(Key) == 8, fmix_64<Key>, murmurhash3_32<Key>>>;

}  // namespace cuco
