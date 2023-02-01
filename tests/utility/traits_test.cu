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

#include <utils.hpp>

#include <cuco/allocator.hpp>
#include <cuco/traits.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <memory>

CUCO_DECLARE_BITWISE_COMPARABLE(float)

struct valid_compound_key {
  int x;
  int y;
};

struct invalid_compound_key {
  int x;
  int y;
  char z;
};

TEST_CASE("Bitwise-comparability tests", "")
{
  SECTION("Type is bitwise comparable.")
  {
    REQUIRE(cuco::is_bitwise_comparable_v<int32_t>);
    REQUIRE(cuco::is_bitwise_comparable_v<int64_t>);
    REQUIRE(cuco::is_bitwise_comparable_v<uint32_t>);
    REQUIRE(cuco::is_bitwise_comparable_v<uint64_t>);
    REQUIRE(cuco::is_bitwise_comparable_v<float>);
    REQUIRE(cuco::is_bitwise_comparable_v<valid_compound_key>);
  }

  SECTION("Type is not bitwise comparable.")
  {
    REQUIRE(not cuco::is_bitwise_comparable_v<double>);
    REQUIRE(not cuco::is_bitwise_comparable_v<invalid_compound_key>);
  }
}

struct not_an_allocator {
};

TEST_CASE("Allocator tests", "")
{
  SECTION("Type is allocator-like.")
  {
    REQUIRE(cuco::detail::is_allocator_v<std::allocator<int>>);
    REQUIRE(cuco::detail::is_allocator_v<cuco::cuda_allocator<int>>);
  }

  SECTION("Type is not allocator-like.")
  {
    REQUIRE(not cuco::detail::is_allocator_v<int>);
    REQUIRE(not cuco::detail::is_allocator_v<not_an_allocator>);
  }
}