# =============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to find or clone warpcore
function(find_and_configure_warpcore)
    set(CPM_DOWNLOAD_VERSION 0.38.1)

    if(CPM_SOURCE_CACHE)
        set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    elseif(DEFINED ENV{CPM_SOURCE_CACHE})
        set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    else()
        set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
    endif()

    # Expand relative path. This is important if the provided path contains a tilde (~)
    get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

    if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
        message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
        file(DOWNLOAD
             https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
             ${CPM_DOWNLOAD_LOCATION}
        )
    endif()

    include(${CPM_DOWNLOAD_LOCATION})

    # Ensure compilers are set for warpcore subproject
    # Warpcore needs these set before its project() call
    set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "C++ compiler" FORCE)

    CPMAddPackage(
        NAME warpcore
        GITHUB_REPOSITORY sleeepyjack/warpcore
        GIT_TAG master
        OPTIONS
            "WARPCORE_BUILD_TESTS OFF"
            "WARPCORE_BUILD_BENCHMARKS OFF"
            "WARPCORE_BUILD_EXAMPLES OFF"
            "CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}"
            "CMAKE_C_COMPILER ${CMAKE_C_COMPILER}"
    )
endfunction()

# Only fetch warpcore if building benchmarks
if(BUILD_BENCHMARKS)
    find_and_configure_warpcore()
endif()
