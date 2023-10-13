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
 */

#pragma once

#include <type_traits>

namespace cuco::detail {

// TODO why is this here?
template <bool value, typename... Args>
inline constexpr bool dependent_bool_value = value;

template <typename... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

}  // namespace cuco::detail