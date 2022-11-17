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
#include <cuco/detail/pair.cuh>
#include <cuco/detail/tags.hpp>
#include <cuco/detail/traits.hpp>

#include <thrust/distance.h>

#include <cooperative_groups.h>
#include <cuda/std/atomic>

namespace cuco {
namespace experimental {
namespace function {  // TODO inline namespace?

class insert {
  enum class insert_result : int32_t { CONTINUE = 0, SUCCESS = 1, DUPLICATE = 2 };

 public:
  template <typename Reference,
            std::enable_if_t<detail::has_tag_v<detail::tag::static_set_ref,
                                               detail::reference_traits<Reference>>>* = nullptr>
  class impl {
    using ref_traits = detail::reference_traits<Reference>;
    using value_type = typename ref_traits::value_type;

    static constexpr auto cg_size     = ref_traits::cg_size;
    static constexpr auto window_size = ref_traits::window_size;
    static constexpr auto scope       = ref_traits::scope;

   public:
    /**
     * @brief Inserts an element.
     *
     * @param value The element to insert
     * @return True if the given element is successfully inserted
     */
    __device__ inline bool insert(value_type const& value) noexcept
    {
      auto probing_iter = ref_.probing_scheme_(value, ref_.storage_ref_.num_windows());

      while (true) {
        auto const window_slots = ref_.storage_ref_.window(*probing_iter);

        for (auto& slot_content : window_slots) {
          auto const eq_res = ref_.predicate_(slot_content, value);

          // If the key is already in the map, return false
          if (eq_res == detail::equal_result::EQUAL) { return false; }
          if (eq_res == detail::equal_result::EMPTY) {
            auto const intra_window_index = thrust::distance(window_slots.begin(), &slot_content);
            switch (attempt_insert(
              (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)) {
              case insert_result::CONTINUE: continue;
              case insert_result::SUCCESS: return true;
              case insert_result::DUPLICATE: return false;
            }
          }
        }
        ++probing_iter;
      }
    }

    /**
     * @brief Inserts an element.
     *
     * @param g The Cooperative Group used to perform group insert
     * @param value The element to insert
     * @return True if the given element is successfully inserted
     */
    __device__ inline bool insert(cooperative_groups::thread_block_tile<cg_size> const& g,
                                  value_type const& value) noexcept
    {
      auto probing_iter = ref_.probing_scheme_(g, value, ref_.storage_ref_.num_windows());

      while (true) {
        auto const window_slots = ref_.storage_ref_.window(*probing_iter);

        auto const [state, intra_window_index] = [&]() {
          for (auto i = 0; i < window_size; ++i) {
            switch (ref_.predicate_(window_slots[i], value)) {
              case detail::equal_result::EMPTY:
                return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EMPTY, i};
              case detail::equal_result::EQUAL:
                return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::EQUAL, i};
              default: continue;
            }
          }
          return cuco::pair<detail::equal_result, int32_t>{detail::equal_result::UNEQUAL, -1};
        }();

        // If the key is already in the map, return false
        if (g.any(state == detail::equal_result::EQUAL)) { return false; }

        auto const group_contains_empty = g.ballot(state == detail::equal_result::EMPTY);

        if (group_contains_empty) {
          auto const src_lane = __ffs(group_contains_empty) - 1;
          auto const status =
            (g.thread_rank() == src_lane)
              ? attempt_insert(
                  (ref_.storage_ref_.windows() + *probing_iter)->data() + intra_window_index, value)
              : insert_result::CONTINUE;

          switch (g.shfl(status, src_lane)) {
            case insert_result::SUCCESS: return true;
            case insert_result::DUPLICATE: return false;
            default: continue;
          }
        } else {
          ++probing_iter;
        }
      }
    }

   private:
    __device__ inline insert_result attempt_insert(value_type* slot, value_type const& key)
    {
      auto slot_ref = cuda::atomic_ref<value_type, scope>{*slot};
      auto expected = ref_.empty_key_sentinel_.value;
      bool result =
        slot_ref.compare_exchange_strong(expected, key, cuda::std::memory_order_relaxed);
      if (result) {
        return insert_result::SUCCESS;
      } else {
        auto old = expected;
        return ref_.predicate_(old, key) == detail::equal_result::EQUAL ? insert_result::DUPLICATE
                                                                        : insert_result::CONTINUE;
      }
    }

    Reference& ref_ = static_cast<Reference&>(*this);  // here comes the CRTP
  };
};

}  // namespace function
}  // namespace experimental
}  // namespace cuco