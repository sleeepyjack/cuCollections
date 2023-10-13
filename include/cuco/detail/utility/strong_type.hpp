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

#include <cuco/detail/utility/cuda.hpp>

namespace cuco::detail {

/**
 * @brief A strong type wrapper.
 *
 * @tparam T Type of the mapped values
 */
template <typename T>
struct strong_type {
  /**
   * @brief Constructs a strong type.
   *
   * @param v Value to be wrapped as a strong type
   */
  CUCO_HOST_DEVICE explicit constexpr strong_type(T v) : value{v} {}

  /**
   * @brief Implicit conversion operator to the underlying value.
   *
   * @return Underlying value
   */
  CUCO_HOST_DEVICE constexpr operator T() const noexcept { return value; }

  T value;  ///< Underlying value
};

}  // namespace cuco::detail