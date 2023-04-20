/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.
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

#include <cstdint>

namespace cuco::detail {

/**
 * @brief The 32bit integer finalizer hash function of `MurmurHash3`.
 *
 * @throw Key type must be 4 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix32 {
  static_assert(sizeof(Key) == 4, "Key type must be 4 bytes in size.");

  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix32(uint32_t seed = 0) : seed_(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    uint32_t h = static_cast<uint32_t>(key) ^ seed_;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

 private:
  uint32_t seed_;
};

/**
 * @brief The 64bit integer finalizer hash function of `MurmurHash3`.
 *
 * @throw Key type must be 8 bytes in size
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_fmix64 {
  static_assert(sizeof(Key) == 8, "Key type must be 8 bytes in size.");

  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_fmix64 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_fmix64(uint64_t seed = 0) : seed_(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    uint64_t h = static_cast<uint64_t>(key) ^ seed_;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
  }

 private:
  uint64_t seed_;
};

/**
 * @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
 *
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 * -----------------------------------------------------------------------------
 * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
 * hereby disclaims copyright to this source code.
 *
 * Note - The x86 and x64 versions do _not_ produce the same results, as the algorithms are
 * optimized for their respective platforms. You can still compile and run any of them on any
 * platform, but your performance with the non-native version will be less than optimal.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_32 {
  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a MurmurHash3_32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_32(uint32_t seed = 0) : fmix32_(0), seed_(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    constexpr int len         = sizeof(argument_type);
    const uint8_t* const data = (const uint8_t*)&key;
    constexpr int nblocks     = len / 4;

    uint32_t h1           = seed_;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];  // getblock32(blocks,i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1         = 0;
    switch (len & 3) {
      case 3: k1 ^= tail[2] << 16;
      case 2: k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32_(h1);
    return h1;
  }

 private:
  constexpr __host__ __device__ uint32_t rotl32(uint32_t x, int8_t r) const noexcept
  {
    return (x << r) | (x >> (32 - r));
  }

  MurmurHash3_fmix32<uint32_t> fmix32_;
  uint32_t seed_;
};

/**
 * @brief A `XXHash_32` hash function to hash the given argument on host and device.
 *
 * XXHash_32 implementation from
 * https://github.com/Cyan4973/xxHash
 * TODO Copyright disclaimer
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct XXHash_32 {
 private:
  static constexpr uint32_t prime1 = 0x9E3779B1U;
  static constexpr uint32_t prime2 = 0x85EBCA77U;
  static constexpr uint32_t prime3 = 0xC2B2AE3DU;
  static constexpr uint32_t prime4 = 0x27D4EB2FU;
  static constexpr uint32_t prime5 = 0x165667B1U;

 public:
  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr XXHash_32(uint32_t seed = 0) : seed_(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    // TODO do we need to add checks/hints for alignment?
    constexpr auto nbytes        = sizeof(Key);
    char const* const bytes      = (char const*)&key;      ///< per-byte access
    uint32_t const* const blocks = (uint32_t const*)&key;  ///< 4-byte word access

    uint32_t offset = 0;
    uint32_t h32;

    // data can be processed in 16-byte chunks
    if constexpr (nbytes >= 16) {
      constexpr auto limit = nbytes - 16;
      uint32_t v1          = seed_ + prime1 + prime2;
      uint32_t v2          = seed_ + prime2;
      uint32_t v3          = seed_;
      uint32_t v4          = seed_ - prime1;

      do {
        // pipeline 4*4byte computations
        auto const pipeline_offset = offset >> 2;  // optimized division by 4
        v1 += blocks[pipeline_offset] * prime2;
        v1 = rotl(v1, 13);
        v1 *= prime1;
        v2 += blocks[pipeline_offset + 1] * prime2;
        v2 = rotl(v2, 13);
        v2 *= prime1;
        v3 += blocks[pipeline_offset + 2] * prime2;
        v3 = rotl(v3, 13);
        v3 *= prime1;
        v4 += blocks[pipeline_offset + 3] * prime2;
        v4 = rotl(v4, 13);
        v4 *= prime1;
        offset += 16;
      } while (offset <= limit);

      h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    } else {
      h32 = seed_ + prime5;
    }

    h32 += nbytes;

    // remaining data can be processed in 4-byte chunks
    if constexpr ((nbytes % 16) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h32 += blocks[offset >> 2] * prime3;  // optimized division by 4
        h32 = rotl(h32, 17) * prime4;
      }
    }

    // the following loop is only needed if the size of the key is no multiple of the block size
    if constexpr (nbytes % 4) {
      while (offset < nbytes) {
        h32 += (bytes[offset] & 255) * prime5;
        h32 = rotl(h32, 11) * prime1;
        ++offset;
      }
    }

    return finalize(h32);
  }

 private:
  constexpr __host__ __device__ uint32_t rotl(uint32_t h, int8_t r) const noexcept
  {
    return ((h << r) | (h >> (32 - r)));
  }

  // avalanche helper
  constexpr __host__ __device__ uint32_t finalize(uint32_t h) const noexcept
  {
    h ^= h >> 15;
    h *= prime2;
    h ^= h >> 13;
    h *= prime3;
    h ^= h >> 16;
    return h;
  }

  uint32_t seed_;
};

/**
 * @brief A `XXHash_64` hash function to hash the given argument on host and device.
 *
 * XXHash_64 implementation from
 * https://github.com/Cyan4973/xxHash and
 * https://github.com/lz4/lz4-java/blob/812d2e27489995645d878c9fb2ddcb83e54cc7d2/src/build/source_templates/xxhash64_hash.template
 * TODO Copyright disclaimer
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct XXHash_64 {
 private:
  static constexpr uint64_t prime1 = 11400714785074694791ULL;
  static constexpr uint64_t prime2 = 14029467366897019727ULL;
  static constexpr uint64_t prime3 = 1609587929392839161ULL;
  static constexpr uint64_t prime4 = 9650029242287828579ULL;
  static constexpr uint64_t prime5 = 2870177450012600261ULL;

 public:
  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint64_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH64 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr XXHash_64(uint64_t seed = 0) : seed_(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    // TODO do we need to add checks/hints for alignment?
    constexpr auto nbytes         = sizeof(Key);
    char const* const bytes       = (char const*)&key;      ///< per-byte access
    uint32_t const* const blocks4 = (uint32_t const*)&key;  ///< 4-byte word access
    uint64_t const* const blocks8 = (uint64_t const*)&key;  ///< 8-byte word access

    uint64_t offset = 0;
    uint64_t h64;

    // data can be processed in 32-byte chunks
    if constexpr (nbytes >= 32) {
      constexpr auto limit = nbytes - 32;
      uint64_t v1          = seed_ + prime1 + prime2;
      uint64_t v2          = seed_ + prime2;
      uint64_t v3          = seed_;
      uint64_t v4          = seed_ - prime1;

      do {
        // pipeline 4*8byte computations
        auto const pipeline_offset = offset >> 3;  // optimized division by 8
        v1 += blocks8[pipeline_offset] * prime2;
        v1 = rotl(v1, 31);
        v1 *= prime1;
        v2 += blocks8[pipeline_offset + 1] * prime2;
        v2 = rotl(v2, 31);
        v2 *= prime1;
        v3 += blocks8[pipeline_offset + 2] * prime2;
        v3 = rotl(v3, 31);
        v3 *= prime1;
        v4 += blocks8[pipeline_offset + 3] * prime2;
        v4 = rotl(v4, 31);
        v4 *= prime1;
        offset += 32;
      } while (offset <= limit);

      h64 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);

      v1 *= prime2;
      v1 = rotl(v1, 31);
      v1 *= prime1;
      h64 ^= v1;
      h64 = h64 * prime1 + prime4;

      v2 *= prime2;
      v2 = rotl(v2, 31);
      v2 *= prime1;
      h64 ^= v2;
      h64 = h64 * prime1 + prime4;

      v3 *= prime2;
      v3 = rotl(v3, 31);
      v3 *= prime1;
      h64 ^= v3;
      h64 = h64 * prime1 + prime4;

      v4 *= prime2;
      v4 = rotl(v4, 31);
      v4 *= prime1;
      h64 ^= v4;
      h64 = h64 * prime1 + prime4;
    } else {
      h64 = seed_ + prime5;
    }

    h64 += nbytes;

    // remaining data can be processed in 8-byte chunks
    if constexpr ((nbytes % 32) >= 8) {
      for (; offset <= nbytes - 8; offset += 8) {
        uint64_t k1 = blocks8[offset >> 3] * prime2;
        k1          = rotl(k1, 31) * prime1;
        h64 ^= k1;
        h64 = rotl(h64, 27) * prime1 + prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if constexpr (((nbytes % 32) % 8) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h64 ^= (blocks4[offset >> 2] & 0xFFFFFFFFULL) * prime1;
        h64 = rotl(h64, 23) * prime2 + prime3;
      }
    }

    // the following loop is only needed if the size of the key is no multiple of a previous block
    // size
    if constexpr (nbytes % 4) {
      while (offset < nbytes) {
        h64 += (bytes[offset] & 0xFF) * prime5;
        h64 = rotl(h64, 11) * prime1;
        ++offset;
      }
    }
    return finalize(h64);
  }

 private:
  constexpr __host__ __device__ uint64_t rotl(uint64_t h, int8_t r) const noexcept
  {
    return ((h << r) | (h >> (64 - r)));
  }

  // avalanche helper
  constexpr __host__ __device__ uint64_t finalize(uint64_t h) const noexcept
  {
    h ^= h >> 33;
    h *= prime2;
    h ^= h >> 29;
    h *= prime3;
    h ^= h >> 32;
    return h;
  }

  uint64_t seed_;
};

}  // namespace cuco::detail
