# cuCollections

<table><tr>
<th><b><a href="https://github.com/NVIDIA/cuCollections/tree/dev/examples">Examples</a></b></th>
<th><b><a href="">Doxygen Documentation (TODO)</a></b></th>
</tr></table>

`cuCollections` (`cuco`) is an open-source, header-only library of GPU-accelerated, concurrent data structures.

Similar to how [Thrust](https://github.com/thrust/thrust) and [CUB](https://github.com/thrust/cub) provide STL-like, GPU-accelerated algorithms and primitives, `cuCollections` provides STL-like concurrent data structures. `cuCollections` is not a one-to-one, drop-in replacement for STL data structures like `std::unordered_map`. Instead, it provides functionally similar data structures optimized for efficient use with GPUs.

## Development Status

`cuCollections` is still under active development. Users should expect breaking changes and refactoring to be common.

### Major Updates

__06/04/2025__ Removed CUDA 11 support

__11/01/2024__ Refined the term `window` as `bucket`

__01/08/2024__ Deprecated the `experimental` namespace

__01/02/2024__ Moved the legacy `static_map` to `cuco::legacy` namespace


## Getting cuCollections

`cuCollections` is header-only and can be incorporated manually into your project by downloading the headers and placing them into your source tree.

### Adding `cuCollections` to a CMake Project

`cuCollections` is designed to make it easy to include within another CMake project.
 The `CMakeLists.txt` exports a `cuco` target that can be linked<sup>[1](#link-footnote)</sup>
 into a target to set up include directories, dependencies, and compile flags necessary to use `cuCollections` in your project.


We recommend using [CMake Package Manager (CPM)](https://github.com/TheLartians/CPM.cmake) to fetch `cuCollections` into your project.
With CPM, getting `cuCollections` is easy:

```cmake
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(path/to/CPM.cmake)

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

target_link_libraries(my_library cuco)
```

This will take care of downloading `cuCollections` from GitHub and making the headers available in a location that can be found by CMake. Linking against the `cuco` target will provide everything needed for `cuco` to be used by the `my_library` target.

<a name="link-footnote">1</a>: `cuCollections` is header-only and therefore there is no binary component to "link" against. The linking terminology comes from CMake's `target_link_libraries` which is still used even for header-only library targets.

## Requirements
- NVCC 12.0 or newer
- C++17
- GPU Architecture: Volta or newer
    - Pascal is partially supported. Any data structures that require blocking algorithms are not supported. See [libcu++](https://nvidia.github.io/libcudacxx/setup/requirements.html#device-architectures) documentation for more details.

## Dependencies

`cuCollections` depends on the following libraries:

- [CUDA C++ Core Libraries (CCCL)](https://github.com/NVIDIA/cccl)

No action is required from the user to satisfy these dependencies. `cuCollections`'s CMake script is configured to first search the system for these libraries, and if they are not found, to automatically fetch them from GitHub.


## Building cuCollections

Since `cuCollections` is header-only, there is nothing to build to use it.

To build the tests, benchmarks, and examples:

```bash
cd $CUCO_ROOT
mkdir -p build
cd build
cmake .. # configure
make # build
ctest --test-dir tests # run tests
```
Binaries will be built into:
- `build/tests/`
- `build/benchmarks/`
- `build/examples/`

### Build Script:

Alternatively, you can use the build script located at `ci/build.sh`. Calling this script with no arguments will trigger a full build which will be located at `build/local`.

```bash
cd $CUCO_ROOT
ci/build.sh # configure and build
ctest --test-dir build/local/tests # run tests
```

For a comprehensive list of all available options along with descriptions and examples, you can use the option `ci/build.sh -h`.

## Code Formatting
By default, `cuCollections` uses [`pre-commit.ci`](https://pre-commit.ci/) along with [`mirrors-clang-format`](https://github.com/pre-commit/mirrors-clang-format) to automatically format the C++/CUDA files in a pull request.
Users should enable the `Allow edits by maintainers` option to get auto-formatting to work.

### Pre-commit hook
Optionally, you may wish to setup a [`pre-commit`](https://pre-commit.com/) hook to automatically run `clang-format` when you make a git commit. This can be done by installing `pre-commit` via `conda` or `pip`:

```bash
conda install -c conda-forge pre_commit
```

```bash
pip install pre-commit
```

and then running:
```bash
pre-commit install
```

from the root of the `cuCollections` repository. Now code formatting will be run each time you commit changes.

You may also wish to manually format the code:
```bash
pre-commit run clang-format --all-files
```

### Caveats
`mirrors-clang-format` guarantees the correct version of `clang-format` and avoids version mismatches.
Users should **_NOT_** use `clang-format` directly on the command line to format the code.


## Documentation
[`Doxygen`](https://doxygen.nl/) is used to generate HTML pages from the C++/CUDA comments in the source code.

### The example
The following example covers most of the Doxygen block comment and tag styles
for documenting C++/CUDA code in `cuCollections`.

```c++
/**
 * @file source_file.cpp
 * @brief Description of source file contents
 *
 * Longer description of the source file contents.
 */

/**
 * @brief Short, one sentence description of the class.
 *
 * Longer, more detailed description of the class.
 *
 * A detailed description must start after a blank line.
 *
 * @tparam T Short description of each template parameter
 * @tparam U Short description of each template parameter
 */
template <typename T, typename U>
class example_class {

  void get_my_int();            ///< Simple members can be documented like this
  void set_my_int( int value ); ///< Try to use descriptive member names

  /**
   * @brief Short, one sentence description of the member function.
   *
   * A more detailed description of what this function does and what
   * its logic does.
   *
   * @param[in]     first  This parameter is an input parameter to the function
   * @param[in,out] second This parameter is used both as an input and output
   * @param[out]    third  This parameter is an output of the function
   *
   * @return The result of the complex function
   */
  T complicated_function(int first, double* second, float* third)
  {
      // Do not use doxygen-style block comments
      // for code logic documentation.
  }

 private:
  int my_int;                ///< An example private member variable
};
```

### Doxygen style check
`cuCollections` also uses Doxygen as a documentation linter. To check the Doxygen style locally, run
```bash
./ci/pre-commit/doxygen.sh
```


## Data Structures

We plan to add many GPU-accelerated, concurrent data structures to `cuCollections`. As of now, the two flagships are variants of hash tables.

### `static_set`

`cuco::static_set` is a fixed-size container that stores unique elements in no particular order. See the Doxygen documentation in `static_set.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNp9VgtvIjcQ_ivTraqSu-UVKTqJPFSapCq6EzmF3J1OpSLGa1grxqZ-wFHEf--MvcvjHk0kYD3jmW9mvs_rbeaEc9Jol_X-2mayyHrdPFNMzwObi6yX8VCwLM-cCZbTc_vVWMMruDXLjZXz0kODn8F55_y8iR8XOQw_Du4Gfbh9eHz_8Nh_GjwMW7QhbnonudBOFBB0ISz4UkB_yTh-VZYcPgpLaOC81YEGOYyzyjbOzi5jlI0JsGAb0MZDcALDSAczqQSIL1wsPUgN3CyWSjLNBaylL2OqKk6EA5-rIGbqGfoz3LHEp9mxJzC_h05_pffLXru9Xq9bLMJuGTtvq-Ts2u8Gt_fD0X0Toe-3fdAK2wtW_BOkxcKnG2BLRMbZFPEqtgZjgc2tQJs3hHxtpZd6noMzM79mVsQ4hXTeymnwJ82rcWL9xw7YPqaxcf0RDEbjDH7vjwajPMb5NHj68-HDE3zqPz72h0-D-xE8POKwhncDGhU-_QH94Wd4Oxje5SCwdZhKfFlaqgKhSmqrKFIPR0KcwJiZBMstBZczyaGmEczNSliNZcFS2IVMhEOQRYyj5EJ65uPaN8XFVO2xHuufpeYqFAKueOCm7WgLnzjhWzyUN197FAw9ivYsaE6Rmbo5tvvSBufbhVhhlslKcG9sq_yeizJznJb6vtHhXAWSrPV1fmlwHIItTjbFOl30bL9K_Pgt8rZE78k0qJeJ-MKwwwIrSuaplWIGd2KBvfGWeYEdctTRiqmHJuC8KQKOm6JB__3AHYT3hK5kjcuApEKOqA3SbWUSE2fWLGLAuBkHE50CaZWmWhiancHhVWNS8kXAs8T5WP-MxIhZnrnRpCb3HCkIhApxvojNAQpOUmqP0pO6sTKyOBvrLa5jKsrwVmzgGmH5S2oSQLsN94ul34BTxifgVhAZhfZJTfTbrvD3OKNFqYXCFqyYCsK1qG7kZXoCV5qgcBN2TSicOG5ygZdVHl8yTx3Ac0UgWcFwHiwJEo8b-l4GDwXzjCqJODmNhKQBgiBOsMxJDQGLaHYPNQzDYkohUy9I6AgiNU-QBHCMRa_n5L9i4o_i6rCYxA3XcNH5tdPpHCLe4vmGWgdOB5HEBk2Zq5SPvr-AMgwnx4jWtIMFzHkITNZJsmLsTuvi8rsgDtGvk5ELqRp7VO3jOGfH2IirgdPAIgfiCYzdVYJh0Oc66nM11TT6ZheYixSM7UydqvsZm06qR4B7xl_hFG4owbaOmFc--4Fsvx3NbneEFCWKTSSUScf7EW07OXRzOM-h1cpB7mhD0nyvd3JoJBC0Z9-Y-Jrae9ehG1EGUzFH4p_lSRRCF_S7c9S8QWQFdk6phATFYJIymSup2jgqPPQSf34Y9ijmCHHSmUFKRr0E5X9cztQYdYOeeA6fFFR3rBT8BeTsgI80WekeCYh6icfSAWZ9JvxP_TFbbTpKhncBOdskZe7zlWyFZ5kQOm0jV8JcMRbdIuRJNCJv6yrJYGaNk1R15goGvTOIX0h0WRBb_Ga7q-FgyY3T4GewrVRh8Gi4usITaBQ4xxflT_BHzF5jbo0xRnYJuxTKCh-sBlIzruDdii4r-BqwhytYplecd88vQhfNZunT_SxrYr5r_vp19w00meXltVtM3nSg2URVe_zwWIkomootpvHSpuT0KCbnXOHiKt2wcAFFql-yXV7bUTsndmRGtvs7_v8HAzB-mg))
- [Device-ref APIs for individual operations](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [One single storage for multiple sets](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/device_subsets_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Using shared memory as storage](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/shared_memory_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Using set as mapping table to handle large keys or indeterministic sentinels](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_set/mapping_table_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))

### `static_map`

`cuco::static_map` is a fixed-size hash table using open addressing with linear probing. See the Doxygen documentation in `static_map.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Device-ref APIs for individual operations](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Custom data types, key equality operators and hash functions](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/custom_type_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Key histogram](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/count_by_key_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))
- [Pre-allocated memory](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_map/preallocated_memory_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))

### `static_multimap`

`cuco::static_multimap` is a fixed-size hash table that supports storing equivalent keys. It uses double hashing by default and supports switching to linear probing. See the Doxygen documentation in `static_multimap.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_multimap/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNqlVgtv2zYQ_isHDUXtVJYfaFDEjQN4bYoZK5whTlsUcaHQFG0TkUmNpOx6hv_77ijJlpsM67AWiCHe-7vvjtwFVlgrtbJB_34XyCTod8MgZWqRs4UI-gHPExaEgdW54fTdPpsqOIN3OtsauVg6aPAm9Dq9bgv_vA5h_Hn0fjSEdze3f9zcDu9GN-OIDLzRR8mFsiKBXCXCgFsKGGaM408pCeGzMJQN9KIONEhhGpSyadB8671sdQ4rtgWlHeRWoBtpYS5TAeI7F5kDqYDrVZZKpriAjXRLH6r049OBr6UTPXMM9RlaZPg1r2sCc4fU6d_Suazfbm82m4j5tCNtFu20ULbtj6N31-PJdQtTP5h9UinCC0b8mUuDhc-2wDLMjLMZ5puyDWgDbGEEypymzDdGOqkWIVg9dxtmhPeTSOuMnOXuBLwqT6y_roDwMYXADScwmkwD-HU4GU1C7-fL6O63m0938GV4ezsc342uJ3Bzi80avx9Rq_DrAwzHX-H30fh9CAKhw1Die2aoCkxVEqwiKTCcCHGSxlwXadlMcDmXHCoawUKvhVFYFmTCrGRBOEwy8X5SuZKOOX_2pDgfqj1VU_WLVDzNEwGXPOe6bcmEx6s8dXLFsojny6tTNbc0uXXtRKzRVbwW3GkTkdITFemEYShtc50rgj-uTp7Xt9hPgeR6XuoMUxbBWEU_ZuQrtf5QKocElKqx1jJpTtUO60Q6E0aPYhu7bSaQcgOkhHt7FK1ZmotCWIlIeLAQq8xtY_q0ggoRqWfuAFpd76VmX6gWBwflUpFUObbDUevBuqTft_IvNIQxqpx3XnY6nVKt3cZdgJom5w5bClU_irnrdjoh6oJNNUqLGlpd6rz_sb7TPhOqoe2zKb1WOdkIxtoVTOM0dxKVkfF8qVEFHpXekNcNTXqaIihWGIc5-rjo1IalQ6InzkWqWQJzRmSgeT_vvIh8ucgprPKUVJcVsGENuSvsW-bbBQjHGfTC0vgA_u5pG_anSt7b7rkW7PclsAWZ-v0T-l4WTjImzfO5XQHJbGPcrPXHCIYAMqhoS3V7tQiuBS4yggmW1I2NxuIcrjYblcbX0SIKYbdDPLGGXTfs4k8URYAnvXMiQnEM_qMU7usFHMahUWBWna_Yo4ifDNwlsvqq0WmGP69clEIEbTQru-JsJhY4YdXZ_TeI4xLOuMFyWrhN2OF2drlR8G_Q7iS8gMYY2tBrhiD3b2FfA3nkeYcTgBz0wWk8tWct0oWUaEkV7GycZlcmK1SC-R89TurdMnomPJsJdyC4iy68vnh5cXGx_2fCVKVceevY6XguMZDnx9GmYkajrnRM8OTU5xlCp84wakwxzJrz3BjyZSlxnzJea_eYdbE3mrRZ6NLFmwoFHqBqh5fu7vDsIdZ4mZkHsPl8Lr-jaoK3phMUhbkfQlEkhg8C1fLspfXQ9c58k_0mA3SX5S4mmuAO8zcGZV2E-em6m_9zOpMYr1Is1zZq-dSQ_ICRChI9bfbe702LgRAFZq3mktF9f9icdeI9HCI9VAwljGjSH-pl3ctvD5BoYdVLhzc9PiPCmi1KIysQwAQGg2evjIcDzGVBgnQ9vjhXRoq1-I8Qh0eQKrWCrD7KwWnZyFrU1lPDI7DEKacdS0Hlq5nwu9_nVS08sEudp_hEQ-6N4RX4QS-qs35o8b3ZOCHR4MdkBnAwPBClXC50XeKU4kuanqb4WDXHB3eg1px3e-d5F8U6c8VrPGjhvTvgr15130CLGb4c2FX8pgOtFl7KDv84rFkkrZStZv6JnspZzSfnPMXDdfGexgO8odVjsA8rOVL1RI5MDvbf_P-_AYKkJA4))

### `static_multiset`

`cuco::static_multiset` is a fixed-size container that supports storing equivalent keys. It uses double hashing by default and supports switching to linear probing. See the Doxygen documentation in `static_multiset.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/static_multiset/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNqVVwtv2zYQ_itXDUPs1s9gRQE3GeYlKWascIo4bVHUhUNTZ5uIJGokZdcL8t93R0q2nMfWJUBi844fv_vuQekusmit0pmNBl_vIhVHg34rSkS2LMQSo0Eki1hErcjqwkj-3n05zeAlnOl8a9Ry5aAhm3DcO_6lBeNPo_PREM4urz5cXg2vR5fjDvt6__dKYmYxhiKL0YBbIQxzIelfaWnBJzRMBI47PWiwwzQqbdOo-dajbHUBqdhCph0UFglGWVioBAG_S8wdqAykTvNEiUwibJRb-aNKHE8HvpQgeu4E-QvakdO3Rd0ThNtR55-Vc_mg291sNh3haXe0WXaT4Gy770dnF-PJRZuo77Z9zBJSFgz-VShDgc-3IHJiJsWc-CZiA9qAWBokm9PMfGOUU9myBVYv3EYY9Dixss6oeeEOxKt4Uvx1B5JPZCTccAKjyTSC34eT0aTlcT6Prv-4_HgNn4dXV8Px9ehiApdXlKzx-YhTRd_ewXD8Bf4cjc9bgCQdHYXfc8NREFXFsmIcNJwgHtBY6EDL5ijVQkmoKgiWeo0mo7AgR5OqUGtEMvY4iUqVE86vPQrOH9WdZtPsJ5XJpIgRTmQhddfyFjlLi8Qpi64ji9Wvh25uZQrrujGuCWq2Rum06bDTI5dFkUk-XyRP2xO9pJQ9Y7SUXKRK6zw8X2nKCYr0YJMP1nrP7stQJL_54l2R92xeJLcz_C5IZqSIgnluFC7gHFMSyBnhkGSyLGtZrg-UoMwzDCWeIWH4YWT3LXhN_mz1y0DlRdWSbKnw1jrU5MLo1KP6zZQi71Rw13J-Y81Z1JTGMmGJukW4UZQp426oRPwpNwapGnGNN74YYUeNGN_ids-HEqsyR52ossZaq7g5ze5onc7jY8h15rY5wikRdG9ZM4BuFy7S3G3BJtqFEAxygWLmQofxZ7Omz9OIF1WGCYmxFkmBtsMKUK2Gb2BXukhoE4mICRUIbbKFXJXnuJVwrAXNGorFgJayMNykNIL4f144iIUTHM6erOQ0cc8AMs8ZGyoeFEm7vw9kXKRzxg2q8AQgJkFL5N6g1MaDgVV_48zVcLMinfkNp_C6d9Tr9faIZzT4aAiA5AmlSKW5sOVIIN-fIdGCEim4FXiHKOjMPTBbZ8FK2L3O67dPktijnwajRJU0dqy6dZxmnRvXbyE5a8DV4EczSZygINCbCvWmTG0ognYfhPUV6eUMSlV6euV5HBDBwy44qfLx66767qoDWuWWXX7uHmfq_r5GnLqYNGXSodV3GbvrtaDfguMWdDotUPflhs94ZHjqMX8_1jmjgHRn8DZwGxpINOBhU7ZfxvP_qCbg8ZGf6DQ13K5hIMybweBgoNXiZMd6Fo79hbnbVpFveMA5Lqnnmq0Aj1nMn3u1bI08aUpVkoRYqQV1mAzCrjh_7LgbvyHGZ7EDlR1smFsiRahNBboIaQ7wzRekqitQdWpowSAgNSLDVBT-L51dVRaZCwXmwQ1rZJlTKpxclXPIcuXzowIl6sG5nWd6hHExnunCUUfO2Ebtsr-u2Pwj_CaUZB72PH1pstH-H6mF8tTc6DnaxhNcDmvjP2C8Es_jlEyvyqHvS-ZAvMNRA18P2M0o4hbMvsEpu9VyWF0i_1KwB0B7l0Pa1XrzmWlWnfMwV49YQvvpA0sJ1AIajWfATp-qh6a_X5_fUvVyswl3QZswbckNTk7oepsUkmrVvoB3BB7vWrUzJUKRj_YeMKHr7unt74RKXsBEp-h8qjY0--gBVGfLOkIIjkgWJgO-amiFXgb4EZueW8z-nSHK1lL2j18XfTLr3IUXiqhNp57KV6_6b6AtjFyd2nT2pgftNl05jv44UhbjdiLSuX_LSNS8himlTGhxHd4LaIFukOw2um9VdprkB3Yq5ej-m__9B3bofi8))

### `dynamic_map`

`cuco::dynamic_map` links together multiple `cuco::static_map`s to provide a hash table that can grow as key-value pairs are inserted. It currently only provides host-bulk APIs. See the Doxygen documentation in `dynamic_map.cuh` for more detailed information.

#### Examples:
- [Host-bulk APIs (TODO)]()

### `hyperloglog`

`cuco::hyperloglog` implements the well-established [HyperLogLog++ algorithm](https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf) for approximating the count of distinct items in a multiset/stream.

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/hyperloglog/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNqNVm1v4kYQ_isj9wvkwAbaKi2XRCUvvVp3IqfAXXQqFVl2B1jFXrv7Akmj_PfOrjExTVrVoES7M_vMM8_MLH6KDBojC2Wi4e9PkRTRsN-JMqZWjq0wGkbcCRZ1IlM4zf06OZopOIKLonzUcrW20OJtGPQGP3Rg_DW9TEdwcX3z-fpmNE2vx7H3Df6fJEdlUIBTAjXYNcKoZJz-7Swd-IraE4FB3IOWd5hFO9ssar8PKI-Fg5w9giosOIMEIw0sZYaADxxLC1IBL_Iyk0xxhK206xBqhxPowLcdSLGwjPwZnShptWx6ArN76v5ZW1sOk2S73cYs0I4LvUqyytkkn9KLq_HkqkvU98e-qIyUBY1_Oqkp8cUjsJKYcbYgvhnbQqGBrTSSzRae-VZLK9WqA6ZY2i3TGHCENFbLhbMH4tU8Kf-mA8nHFAk3mkA6mUVwPpqkk07AuU2nv11_mcLt6OZmNJ6mVxO4vqFijS9TXypa_Qqj8Tf4mI4vO4AkHYXCh1L7LIiq9LKiqDScIB7QWBYVLVMil0vJoe4gWBUb1IrSghJ1LqteI5Ii4GQyl5bZsPcquRAqmanvpOKZEwgn3PEiWT8SUlas6Btztz6bqaaLXWtnbCJwQyDzDXJb6Ng7vXIxVBmkNon_CcFzZg9PcGOFwOXBnixIdmR5OJwcVUX_JTTjmkzzhcvu5_jASDYknpV5oSUu4RJzSthqZpHSNl4mar87n91w2EjvjirpYaiQHhJGn1NTiyKVpS6WqrUppGjP1BPtE5YXegr_9pxSm1k_StTzRMBXFyi14dDIv3BuQbl8Li3mhjz7Lsvg5AQGP70HSBLofzj3mQJU8g2HBxKfTM8gnGztMfzMen86-wEV-mzhbm-9C31LatrqXBO5rkwrWOIFrijPdqdyjFEJv-g18G8RqDxSVU0p0ISJM5bajGkBnmlosnrKaRB18SDzahO1LvQOqRf3-oNB_-djP1kV2JK5zMKGZQ5935J0mmaiLJQwfnAZfD_4eA7mHi1fgxfSQzFHpiAyGEFqVrWtGc33jJ72AZ9f0kkVXQQsI6hAAUknolpxfNUkXvm9x5MRDZyREMCoilVJidArtP0iZkL8h9ztA1DfZGHaWY47cLbyd-lWUjR_ObPlktoiOJFWpN__DLZbwrtGKyYwaIS_YBl3me8lD85JTKlIKvtYwwf5m01dVaG2innzzGmDVO3RqqMFEF4468dgFk21Owg4pD1v2TP1Z16ecGY2U1d14LfOvs1qf9Q3pnd9hRyosYVpHVrCY_yFygnQ2BNROPq1OWu9GadN0r7l_DLB0IV-3Gu_HZ4aI9sppdE6raBHy2d6WfA_wXQP6pd3ikhtOO8PfnR9MhelrV44oi4BnfJ37_rH0GWar09NPj_uQbdLN5OlP5ZioOhmLF-Et5BMLhqYnPOMNjfVewNtWO3UffTcqe00Kgd2mrro-Y_w-Rvi8QMW))
- [Device-ref APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/hyperloglog/device_ref_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))

### `bloom_filter`

`cuco::bloom_filter` implements a Blocked Bloom Filter for approximate set membership queries.

#### Examples:
- [Host-bulk APIs (Default fingerprinting policy)](https://github.com/NVIDIA/cuCollections/blob/dev/examples/bloom_filter/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNqdVmtvGjkU_StXsx8WmuEVbVUJQiSapLtoK5IF2qpaVsjj8TBWBnvqBwRF-e977ZmBIZBVtVRqwL6Pc889vvZzoJnWXAod9P9-Dngc9HthkBGxsmTFgn5AbUyCMNDSKup-d94tBLyDG5nvFF-lBhq0CZfdy99CmHwd345HcHM_fbifjubj-0nb2Xr7z5wyoVkMVsRMgUkZjHJC8U-5E8JXphwQuGx3oeEMFkG5twiaAx9lJy2syQ6ENGA1wzBcQ8IzBuyJstwAF0DlOs84EZTBlpvUpyrjeDjwvQwiI0PQnqBHjr-SuiUQs4fuPqkxeb_T2W63beJht6VadbLCWHc-j2_uJrO7FkLfu30RGTILiv2wXGHh0Q5IjsgoiRBvRrYgFZCVYrhnpEO-VdxwsQpBy8RsiWI-Tsy1UTyy5oi8CifWXzdA-ohA4kYzGM8WAXwczcaz0Mf5Np7_cf9lDt9G0-loMh_fzeB-is2a3I5dq_DXJxhNvsOf48ltCAypw1TsKVeuCoTKHa0sLjicMXYEI5EFLJ0zyhNOoVIQrOSGKYFlQc7UmhdaQ5Cxj5PxNTfE-LWT4nyqzkIsxC9c0MzGDK6opbITZVKul9h3w1Sb2vT62MakymrTodIK03abJ1sx22CK5YZRI9V5E_bEqHXAlrnEpu3OW2nsLkOptV9j4BKbwsjaL3NhUHFcNDaSx82FeMbCwC1SrNs4jkHY9fKR7bQT2xB63V-73e4A9p9Op3MFvzPBFDGs3AZnfz6SyQu34SHuO-i23w_KSGNkVxnPdcKVNpCSLPHxXDBZbnh630ggThK0ysyDUw9to7JbGj0ufWUeB351m61qEyUCvrlldkeeM3XQwOzyQmjHFoCTwMkLTZbeZOiyD_aus5xs3SmvexWTIWYJsZmBosFOk68B-eqd5Pr9uuauqlzXZbznWoUvZWpt4n4fNWjg6gpP5EebPSIwz7vn-G08CZbDVK4ciaui524uFjD7GGshoPbB8D4XE3FW5i7U2e8f6byG2jWsUXXOz9a9S6XohttqR2yFsm2G3qONGdz3XrNMQ6zTSr70Vsh73WVQN0BHL5a97UVdLIWV2IcpHI62qgAHGIcGnxdzOc-9gzMs5UziuFGhCMtMzf9kLZIyu3aWOAixO40CeAgJyTQ75u6so6g7Hp2UWoyylL8sU7va6dvPVZzCGy6tznalhnDW70tDx7m7CHUqbRZDkQ781WaUZa1carxYNgz88EBi5g_TYa_OCh5VdxfqE2rCQ-FVY38Gri_TpMQA3mL-tnbXCBMe1avh4o4od7d1_UBo75grGZEMrze8UGJiCOpcWWosxgprYcoo7CnlETfoWvD6qu5PD9PrLp6xHMty80IWSLAVESJHVrwvr2qrVeUGw7FpSnTKtHuCxH5mYb3n6RR7OkVJp3iDziSTSJdj24EdFkfcLzYqefkLrXEstjMNqi-VJ9bJoNmEThmwkF-h3SJv8j_yitO84ifyiub5AelEiS8W_F6R4Jdd38r15LD-euAphqoQ0MWfL_hadW9AfBWqw6M2EBtKe5fvbQ-3ZW6KF2_QwkBDenHR-wAtomg61Ovlhy60WnhvGfzPYA4WtzKyjvwzOONRLSalNMPFTfFwxQWsVzwGL2G1jzfH0T5yF7z84__9C94J7HA))

### roaring_bitmap

`cuco::experimental::roaring_bitmap` implements a Roaring bitmap following the [Roaring bitmap format specification](https://github.com/RoaringBitmap/RoaringFormatSpec).

#### Examples:
- [Host-bulk APIs](https://github.com/NVIDIA/cuCollections/blob/dev/examples/roaring_bitmap/host_bulk_example.cu) (see [live example in godbolt](https://godbolt.org/clientstate/eNpFjs0KwjAQhF9l2asNNYIIhV49eRDBk4rEzVoDaVLyU5XiuxsRkYWBndkdvgkjx2i8i9gcJjQaG1mhVa7LqmNskLJWWGH0OdBnr2tYZ2vhawB5zZC8B6tCx3D1Afa7zdGVs61lFRlGw3dIN_59XI1l0CYwJfsszeT7oVjhD4BuJJKLZZYl9kP60qGISbc0m8kVCBXo1sb-vJqDEPwYUpHETrMWVvWXgvw6lXkDPzNKTw))