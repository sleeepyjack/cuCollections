/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <type_traits>
#include <utility>

namespace cuco {
namespace detail {

template <class T,
          class V  = typename T::value_type,
          class P  = V*,
          typename = decltype(
            std::declval<T const&>() == T{std::declval<T const&>()},
            std::declval<T&>().deallocate(P{std::declval<T&>().allocate(std::declval<int>())},
                                          std::declval<int>()))>
std::true_type is_allocator_impl(T const&);
std::false_type is_allocator_impl(...);

template <class T>
struct is_allocator : decltype(is_allocator_impl(std::declval<T const&>())) {
};

template <typename A>
inline constexpr bool is_allocator_v = is_allocator<A>::value;

}  // namespace detail
}  // namespace cuco