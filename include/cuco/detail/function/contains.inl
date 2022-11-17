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

#include <cuco/detail/__config>
#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/tags.hpp>
#include <cuco/detail/traits.hpp>

#include <cooperative_groups.h>
#include <cuda/std/atomic>

namespace cuco {
namespace experimental {
namespace function {  // TODO inline namespace?

class contains {
 public:
  template <typename Reference,
            std::enable_if_t<detail::has_tag_v<detail::tag::static_set_ref,
                                               detail::reference_traits<Reference>>>* = nullptr>
  class impl {
    static constexpr auto cg_size = detail::reference_traits<Reference>::cg_size;

   public:
    /**
     * @brief Indicates whether the probe key `key` was inserted into the map/set.
     *
     * If the probe key `key` was inserted into the map/set, returns
     * true. Otherwise, returns false.
     *
     * @tparam ProbeKey Probe key type
     *
     * @param key The key to search for
     * @return A boolean indicating whether the probe key was inserted
     */
    template <typename ProbeKey>
    __device__ inline bool contains(ProbeKey const& key) const noexcept
    {
      auto probing_iter = probing_scheme_(key, ref_.storage_ref_.num_windows());

      while (true) {
        auto const window_slots = ref_.storage_ref_.window(*probing_iter);

        for (auto& slot_content : window_slots) {
          switch (ref_.predicate_(slot_content, key)) {
            case detail::equal_result::UNEQUAL: continue;
            case detail::equal_result::EMPTY: return false;
            case detail::equal_result::EQUAL: return true;
          }
        }
        ++probing_iter;
      }
    }

    /**
     * @brief Indicates whether the probe key `key` was inserted into the map/set.
     *
     * If the probe key `key` was inserted into the map/set, returns
     * true. Otherwise, returns false.
     *
     * @tparam Key Probe key type
     *
     * @param g The Cooperative Group used to perform group contains
     * @param key The key to search for
     * @return A boolean indicating whether the probe key was inserted
     */
    template <typename Key>
    __device__ inline bool contains(cooperative_groups::thread_block_tile<cg_size> const& g,
                                    Key const& key) const noexcept
    {
      auto probing_iter = ref_.probing_scheme_(g, key, ref_.storage_ref_.num_windows());

      while (true) {
        auto const window_slots = ref_.storage_ref_.window(*probing_iter);

        auto const state = [&]() {
          for (auto& slot : window_slots) {
            switch (ref_.predicate_(slot, key)) {
              case detail::equal_result::EMPTY: return detail::equal_result::EMPTY;
              case detail::equal_result::EQUAL: return detail::equal_result::EQUAL;
              default: continue;
            }
          }
          return detail::equal_result::UNEQUAL;
        }();

        if (g.any(state == detail::equal_result::EQUAL)) { return true; }
        if (g.any(state == detail::equal_result::EMPTY)) { return false; }

        ++probing_iter;
      }
    }

   private:
    Reference const& ref_ = static_cast<Reference const&>(*this);  // here comes the CRTP
  };
};

}  // namespace function
}  // namespace experimental
}  // namespace cuco