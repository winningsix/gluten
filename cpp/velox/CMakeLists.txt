# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

project(velox)

include(ExternalProject)
include(FindPkgConfig)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

set(SYSTEM_LIB_PATH
    "/usr/lib"
    CACHE PATH "System Lib dir")
set(SYSTEM_LIB64_PATH
    "/usr/lib64"
    CACHE PATH "System Lib64 dir")
set(SYSTEM_LOCAL_LIB_PATH
    "/usr/local/lib"
    CACHE PATH "System Local Lib dir")
set(SYSTEM_LOCAL_LIB64_PATH
    "/usr/local/lib64"
    CACHE PATH "System Local Lib64 dir")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set(SYSTEM_LIB_MULTIARCH_PATH
      "/usr/lib/x86_64-linux-gnu"
      CACHE PATH "System Lib MultiArch dir")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
  set(SYSTEM_LIB_MULTIARCH_PATH
      "/usr/lib/aarch64-linux-gnu"
      CACHE PATH "System Lib MultiArch dir")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
  set(SYSTEM_LIB_MULTIARCH_PATH
      "/usr/lib"
      CACHE PATH "System Lib MultiArch dir")
else()
  message(FATAL_ERROR "Unsupported processor type: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT DEFINED VELOX_HOME)
  set(VELOX_HOME ${GLUTEN_HOME}/ep/build-velox/build/velox_ep)
  message(STATUS "Set VELOX_HOME to ${VELOX_HOME}")
endif()

message("Velox module final CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

# User can specify VELOX_BUILD_PATH, if Velox are built elsewhere.
if(NOT DEFINED VELOX_BUILD_PATH)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(VELOX_BUILD_PATH
        "${VELOX_HOME}/_build/debug"
        CACHE PATH "Velox build directory.")
  else()
    set(VELOX_BUILD_PATH
        "${VELOX_HOME}/_build/release"
        CACHE PATH "Velox build directory.")
  endif()
endif()

set(VELOX_PROTO_SRC_DIR
    ${GLUTEN_HOME}/backends-velox/src/main/resources/org/apache/gluten/proto)
message(STATUS "Set Gluten Proto Directory in ${VELOX_PROTO_SRC_DIR}")

function(import_library TARGET_NAME LIB_PATH)
  if(NOT EXISTS ${LIB_PATH})
    message(FATAL_ERROR "Library does not exist: ${LIB_PATH}")
  endif()
  add_library(${TARGET_NAME} STATIC IMPORTED)
  set_target_properties(${TARGET_NAME} PROPERTIES IMPORTED_LOCATION ${LIB_PATH})
endfunction()

macro(add_duckdb)
  find_package(DuckDB)
  if(NOT DuckDB_FOUND)
    message(FATAL_ERROR "Cannot find DuckDB.")
  else()
    message(STATUS "Found DuckDB library from ${DuckDB_DIR}")
    target_link_libraries(facebook::velox INTERFACE duckdb_static)
  endif()
endmacro()

macro(find_re2)
  find_package(re2 CONFIG)
  if(re2_FOUND AND TARGET re2::re2)
    set(RE2_LIBRARY re2::re2)
  else()
    find_library(
      RE2_LIBRARY
      NAMES re2
      PATHS ${SYSTEM_LIB_PATH} ${SYSTEM_LIB64_PATH} ${SYSTEM_LIB_MULTIARCH_PATH}
            ${SYSTEM_LOCAL_LIB_PATH} ${SYSTEM_LOCAL_LIB64_PATH}
      NO_DEFAULT_PATH)
  endif()

  if(NOT RE2_LIBRARY)
    message(FATAL_ERROR "RE2 Library Not Found")
  else()
    message(STATUS "RE2 Library Can Be Found in ${RE2_LIBRARY}")
  endif()
endmacro()

macro(find_awssdk)
  find_package(AWSSDK REQUIRED COMPONENTS s3;identity-management;s3-crt)
endmacro()

macro(find_gcssdk)
  find_package(google_cloud_cpp_storage CONFIG 2.22.0 REQUIRED)
endmacro()

macro(find_azure)
  find_package(CURL REQUIRED)
  find_package(LibXml2 REQUIRED)
  find_package(azure-storage-blobs-cpp CONFIG REQUIRED)
  find_package(azure-storage-files-datalake-cpp CONFIG REQUIRED)
  find_package(azure-identity-cpp CONFIG REQUIRED)
endmacro()

# Set up Proto
set(PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto)

# List Gluten Proto compiled files
file(GLOB VELOX_PROTO_FILES ${VELOX_PROTO_SRC_DIR}/*.proto)
foreach(PROTO ${VELOX_PROTO_FILES})
  file(RELATIVE_PATH REL_PROTO ${VELOX_PROTO_SRC_DIR} ${PROTO})
  string(REGEX REPLACE "\\.proto" "" PROTO_NAME ${REL_PROTO})
  list(APPEND VELOX_PROTO_SRCS "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.cc")
  list(APPEND VELOX_PROTO_HDRS "${PROTO_OUTPUT_DIR}/${PROTO_NAME}.pb.h")
endforeach()
set(VELOX_PROTO_OUTPUT_FILES ${VELOX_PROTO_HDRS} ${VELOX_PROTO_SRCS})
set_source_files_properties(${VELOX_PROTO_OUTPUT_FILES} PROPERTIES GENERATED
                                                                   TRUE)
get_filename_component(VELOX_PROTO_DIR ${VELOX_PROTO_SRC_DIR}/ DIRECTORY)

# Build Velox backend.
set(VELOX_SRCS
    ${VELOX_PROTO_SRCS}
    compute/FluxDumpLoader.cc
    compute/FluxQueryCoordinator.cc
    compute/VeloxBackend.cc
    compute/VeloxRuntime.cc
    compute/VeloxPlanConverter.cc
    compute/WholeStageResultIterator.cc
    compute/iceberg/IcebergPlanConverter.cc
    jni/JniFileSystem.cc
    jni/JniUdf.cc
    jni/FluxJniWrapper.cc
    jni/VeloxJniWrapper.cc
    memory/BufferOutputStream.cc
    memory/VeloxColumnarBatch.cc
    memory/VeloxMemoryManager.cc
    operators/functions/CheckOverflowInTableInsert.cc
    operators/functions/RegistrationAllFunctions.cc
    operators/functions/RowConstructorWithNull.cc
    operators/functions/SparkExprToSubfieldFilterParser.cc
    operators/plannodes/RowVectorStream.cc
    operators/reader/FileReaderIterator.cc
    operators/reader/ParquetReaderIterator.cc
    operators/serializer/VeloxColumnarBatchSerializer.cc
    operators/serializer/VeloxColumnarToRowConverter.cc
    operators/serializer/VeloxRowToColumnarConverter.cc
    operators/writer/VeloxColumnarBatchWriter.cc
    operators/writer/VeloxParquetDataSource.cc
    shuffle/ArrowShuffleDictionaryWriter.cc
    shuffle/VeloxHashShuffleWriter.cc
    shuffle/VeloxRssSortShuffleWriter.cc
    shuffle/VeloxShuffleReader.cc
    shuffle/VeloxShuffleWriter.cc
    shuffle/VeloxSortShuffleWriter.cc
    substrait/SubstraitExtensionCollector.cc
    substrait/SubstraitParser.cc
    substrait/SubstraitToVeloxExpr.cc
    substrait/SubstraitToVeloxPlan.cc
    substrait/SubstraitToVeloxPlanValidator.cc
    substrait/VariantToVectorConverter.cc
    substrait/VeloxSubstraitSignature.cc
    substrait/VeloxToSubstraitExpr.cc
    substrait/VeloxToSubstraitPlan.cc
    substrait/VeloxToSubstraitType.cc
    udf/UdfLoader.cc
    utils/ArrowAbiCompat.cc
    utils/Common.cc
    utils/ConfigExtractor.cc
    utils/VeloxArrowUtils.cc
    utils/VeloxBatchResizer.cc
    utils/VeloxWholeStageDumper.cc
    utils/VeloxWriterUtils.cc)

# AArch64 libstdc++ provides std::to_chars(_Float128) directly and GCC does not
# ship libquadmath on this architecture. The shim is only needed on platforms
# where that overload is absent.
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  list(APPEND VELOX_SRCS utils/GccFloat128ToCharsShim.cc)
endif()

if(ENABLE_S3)
  find_package(ZLIB)
endif()

if(ENABLE_GPU)
  list(
    APPEND
    VELOX_SRCS
    cudf/CheckOverflowInTableInsertCudf.cc
    cudf/CudfPlanValidator.cc
    cudf/GpuLock.cc
    cudf/GpuMemoryTracker.cc
    operators/plannodes/CudfVectorStream.cc
    shuffle/VeloxGpuShuffleReader.cc
    shuffle/VeloxGpuShuffleWriter.cc
    operators/serializer/VeloxGpuColumnarBatchSerializer.cc
    utils/GpuBufferBatchResizer.cc
    memory/GpuBufferColumnarBatch.cc
    compute/iceberg/CudfIcebergWriter.cc)
endif()

# Iceberg V2 data writing is an independent OSS capability.  Keep it available
# in regular builds instead of coupling it to the unrelated enhanced-features
# bundle; the Spark-side Iceberg component still controls whether it is used.
list(APPEND VELOX_SRCS compute/iceberg/IcebergFormat.cc
     compute/iceberg/IcebergWriter.cc)

if(BUILD_TESTS OR BUILD_BENCHMARKS)
  set(BUILD_TEST_UTILS ON)
endif()

if(BUILD_TEST_UTILS)
  list(APPEND VELOX_SRCS tests/utils/LocalRssClient.cc
       tests/utils/TestAllocationListener.cc)
endif()

add_library(velox SHARED ${VELOX_SRCS})

# Produce libgluten.so instead of libvelox.so so that a single shared library
# contains both core Gluten JNI symbols and backend (Velox/FLUX) JNI symbols.
set_target_properties(velox PROPERTIES OUTPUT_NAME gluten)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  # Export JNI entry points and Gluten/Velox C++ symbols; hide everything else.
  target_link_options(
    velox PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/symbols.map)
endif()

find_protobuf()
message(STATUS "Found Protobuf: ${PROTOBUF_LIBRARY}")

add_custom_command(
  OUTPUT ${VELOX_PROTO_OUTPUT_FILES}
  COMMAND ${PROTOC_BIN} --proto_path ${VELOX_PROTO_SRC_DIR}/ --cpp_out
          ${PROTO_OUTPUT_DIR} ${VELOX_PROTO_FILES}
  DEPENDS ${VELOX_PROTO_DIR}
  COMMENT "Running Gluten Velox PROTO compiler"
  VERBATIM)

add_custom_target(velox_jni_proto ALL DEPENDS ${SUBSTRAIT_PROTO_OUTPUT_FILES}
                                              ${VELOX_PROTO_OUTPUT_FILES})
add_dependencies(velox_jni_proto protobuf::libprotobuf)

add_dependencies(velox velox_jni_proto)

set(GLUTEN_UCX_INCLUDE_DIR "")
set(GLUTEN_UCX_LIBRARIES "")
if(DEFINED GLUTEN_UCX_HOME AND NOT "${GLUTEN_UCX_HOME}" STREQUAL "")
  # Accept either a standard UCX installation prefix (include/ + lib/) or the
  # legacy flat directory containing both headers and shared libraries.
  if(EXISTS "${GLUTEN_UCX_HOME}/include/ucp/api/ucp.h")
    set(GLUTEN_UCX_INCLUDE_DIR ${GLUTEN_UCX_HOME}/include)
    set(GLUTEN_UCX_LIBRARY_DIR ${GLUTEN_UCX_HOME}/lib)
  else()
    set(GLUTEN_UCX_INCLUDE_DIR ${GLUTEN_UCX_HOME})
    set(GLUTEN_UCX_LIBRARY_DIR ${GLUTEN_UCX_HOME})
  endif()
  set(GLUTEN_UCX_LIBRARIES
      ${GLUTEN_UCX_LIBRARY_DIR}/libucp.so ${GLUTEN_UCX_LIBRARY_DIR}/libuct.so
      ${GLUTEN_UCX_LIBRARY_DIR}/libucs.so)
  if(EXISTS "${GLUTEN_UCX_LIBRARY_DIR}/libucm.so")
    list(APPEND GLUTEN_UCX_LIBRARIES ${GLUTEN_UCX_LIBRARY_DIR}/libucm.so)
  endif()
else()
  find_path(GLUTEN_UCX_INCLUDE_DIR NAMES ucp/api/ucp.h REQUIRED)
  find_library(GLUTEN_UCP_LIBRARY NAMES ucp REQUIRED)
  find_library(GLUTEN_UCT_LIBRARY NAMES uct REQUIRED)
  find_library(GLUTEN_UCS_LIBRARY NAMES ucs REQUIRED)
  list(APPEND GLUTEN_UCX_LIBRARIES ${GLUTEN_UCP_LIBRARY} ${GLUTEN_UCT_LIBRARY}
       ${GLUTEN_UCS_LIBRARY})
  find_library(GLUTEN_UCM_LIBRARY NAMES ucm)
  if(GLUTEN_UCM_LIBRARY)
    list(APPEND GLUTEN_UCX_LIBRARIES ${GLUTEN_UCM_LIBRARY})
  endif()
endif()

set(GLUTEN_CUDF_BUILD_TREE_INCLUDE_DIRS)
if(ENABLE_GPU AND cudf_SOURCE STREQUAL "BUNDLED")
  list(
    APPEND
    GLUTEN_CUDF_BUILD_TREE_INCLUDE_DIRS
    ${VELOX_BUILD_PATH}/_deps/cudf-src/cpp/include
    ${VELOX_BUILD_PATH}/_deps/rmm-src/cpp/include
    ${VELOX_BUILD_PATH}/_deps/rmm-build/include
    ${VELOX_BUILD_PATH}/_deps/cccl-src/libcudacxx/include
    ${VELOX_BUILD_PATH}/_deps/cccl-src/cub
    ${VELOX_BUILD_PATH}/_deps/cccl-src/thrust
    ${VELOX_BUILD_PATH}/_deps/cuco-src/include
    ${VELOX_BUILD_PATH}/_deps/kvikio-src/cpp/include
    ${VELOX_BUILD_PATH}/_deps/rapids_logger-src/include
    /opt/velox/_build/ucx-probe/_deps/ucxx-src/cpp/include
    ${VELOX_BUILD_PATH}/_deps/ucxx-src/cpp/include)
endif()

target_include_directories(
  velox
  PUBLIC ${CMAKE_SYSTEM_INCLUDE_PATH}
         ${JNI_INCLUDE_DIRS}
         ${VELOX_BUILD_PATH}
         # BUNDLED cuDF and UCXX do not propagate every FetchContent include
         # path through their Velox targets. SYSTEM uses only installed package
         # targets and must not inherit these build-tree paths.
         ${GLUTEN_CUDF_BUILD_TREE_INCLUDE_DIRS}
         ${VELOX_BUILD_PATH}/_deps/fmt-src/include
         ${VELOX_BUILD_PATH}/_deps/xsimd-src/include
         ${VELOX_BUILD_PATH}/_deps/re2-src
         ${VELOX_BUILD_PATH}/_deps/absl-src
         # CUDA runtime headers (cuda_runtime_api.h etc.) used transitively by
         # rmm/cuda_stream_view.hpp.
         /usr/local/cuda/include
         ${GLUTEN_UCX_INCLUDE_DIR}
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${VELOX_HOME}
         ${PROTO_OUTPUT_DIR}
         ${PROTOBUF_INCLUDE}
         # Forward gluten's public include dirs (substrait/*.pb.h etc.) to
         # downstream consumers. Previously propagated transitively via
         # `target_link_libraries(velox PUBLIC ... gluten ...)`, which now only
         # lives in velox's PRIVATE scope to avoid baking libgluten.a into
         # consumers a second time.
         $<TARGET_PROPERTY:gluten,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(velox PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                       ${root_directory}/releases)

# If folly is not installed in system lib paths, please add
# `-DCMAKE_PREFIX_PATH="${folly lib path}" to cmake arguments. It is also
# applicable to other dependencies.
find_package(Folly REQUIRED CONFIG)

# Some Folly package configurations expose only Folly's own include directory
# and omit headers from these public dependencies. Discover them explicitly so a
# separately installed Folly can be used with a separately built Velox tree.
find_path(GLUTEN_DOUBLE_CONVERSION_INCLUDE_DIR
          NAMES double-conversion/double-conversion.h REQUIRED)
find_path(GLUTEN_LIBEVENT_INCLUDE_DIR NAMES event2/event.h REQUIRED)
find_path(GLUTEN_FBTHRIFT_INCLUDE_DIR
          NAMES thrift/lib/cpp2/protocol/CompactV1Protocol.h REQUIRED)
target_include_directories(
  velox PUBLIC ${GLUTEN_DOUBLE_CONVERSION_INCLUDE_DIR}
               ${GLUTEN_LIBEVENT_INCLUDE_DIR} ${GLUTEN_FBTHRIFT_INCLUDE_DIR})

if(ENABLE_JEMALLOC_STATS)
  include(Findjemalloc)
  find_jemalloc()
  if(JEMALLOC_NOT_FOUND)
    include(Buildjemalloc)
    build_jemalloc()
  endif()
  add_definitions(-DENABLE_JEMALLOC_STATS)
  target_link_libraries(velox PUBLIC jemalloc::jemalloc)
endif()

# Link the static gluten core library with --whole-archive so that JNI entry
# points (referenced only from Java, not from C++) are not dropped by the
# linker. Keep this PRIVATE: the whole-archive/allow-multiple-definition dance
# is only needed for libgluten.so's own build. Propagating it via PUBLIC causes
# every downstream consumer (executables, plugin .so's) to bake libgluten.a +
# libprotobuf.a into themselves a second time, which FATALs at runtime when
# protobuf's global descriptor pool detects the same .proto registered twice.
target_link_libraries(
  velox PRIVATE -Wl,--whole-archive gluten -Wl,--no-whole-archive
                -Wl,--allow-multiple-definition)

# Requires VELOX_MONO_LIBRARY=ON when building Velox.
import_library(facebook::velox ${VELOX_BUILD_PATH}/lib/libvelox.a)

# velox/tpch/gen generator functions (mk_region, sd_cust, etc.) live in
# libdbgen.a. When libvelox.a is whole-archive'd into libgluten.so, velox's tpch
# code is pulled in and needs these symbols regardless of BUILD_TESTS.
import_library(facebook::velox::dbgen
               ${VELOX_BUILD_PATH}/velox/tpch/gen/dbgen/libdbgen.a)
add_duckdb()

import_library(
  facebook::velox::dwio_parquet_thrift_raw
  ${VELOX_BUILD_PATH}/velox/dwio/parquet/thrift/libvelox_dwio_parquet_thrift_raw.a
)
set(GLUTEN_S2_LIBRARY "${VELOX_BUILD_PATH}/_deps/s2geometry-build/libs2.a")
if(NOT EXISTS "${GLUTEN_S2_LIBRARY}")
  # Velox's setup script may install s2geometry as a system dependency. In that
  # case the Velox build consumes /usr/local/lib/libs2.a and no bundled
  # s2geometry-build directory is created.
  unset(GLUTEN_S2_LIBRARY)
  find_library(GLUTEN_S2_LIBRARY NAMES s2 REQUIRED)
endif()
import_library(external::s2 ${GLUTEN_S2_LIBRARY})
file(GLOB GLUTEN_ABSL_LIBRARIES "/usr/local/lib/libabsl*.a"
     "/usr/local/lib64/libabsl*.a"
     "${VELOX_BUILD_PATH}/_deps/absl-build/absl/*/libabsl*.a")
list(SORT GLUTEN_ABSL_LIBRARIES)
set(GLUTEN_QUADMATH_LIBRARY "")
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(GLUTEN_QUADMATH_LIBRARY quadmath)
endif()

if(BUILD_TESTS)
  target_link_libraries(facebook::velox INTERFACE facebook::velox::dbgen)

  import_library(
    facebook::velox::vector_test_lib
    ${VELOX_BUILD_PATH}/velox/vector/tests/utils/libvelox_vector_test_lib.a)
  import_library(
    facebook::velox::dwio_common_test
    ${VELOX_BUILD_PATH}/velox/dwio/common/tests/utils/libvelox_dwio_common_test_utils.a
  )
  import_library(
    facebook::velox::file_test_utils
    ${VELOX_BUILD_PATH}/velox/common/file/tests/libvelox_file_test_utils.a)
  set(VELOX_TEMP_PATH_LIBRARY
      ${VELOX_BUILD_PATH}/velox/exec/tests/utils/libvelox_temp_path.a)
  set(VELOX_EXEC_TEST_UTILS_CMAKE
      ${VELOX_HOME}/velox/exec/tests/utils/CMakeLists.txt)
  set(VELOX_TEMP_PATH_INTERFACE_DEFINITION "")
  if(EXISTS "${VELOX_EXEC_TEST_UTILS_CMAKE}")
    file(STRINGS "${VELOX_EXEC_TEST_UTILS_CMAKE}"
         VELOX_TEMP_PATH_INTERFACE_DEFINITION
         REGEX "velox_add_library\\(velox_temp_path[ \\t]+INTERFACE")
  endif()
  if(VELOX_TEMP_PATH_INTERFACE_DEFINITION)
    foreach(VELOX_TEMP_PATH_HEADER TempDirectoryPath.h TempFilePath.h)
      if(NOT EXISTS
         "${VELOX_HOME}/velox/exec/tests/utils/${VELOX_TEMP_PATH_HEADER}")
        message(
          FATAL_ERROR
            "Velox header-only temp_path is missing ${VELOX_TEMP_PATH_HEADER}")
      endif()
    endforeach()
    add_library(facebook::velox::temp_path INTERFACE IMPORTED)
    set_target_properties(
      facebook::velox::temp_path PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                            "${VELOX_HOME}")
    message(STATUS "Importing Velox temp_path as a header-only target")
  elseif(EXISTS "${VELOX_TEMP_PATH_LIBRARY}")
    import_library(facebook::velox::temp_path ${VELOX_TEMP_PATH_LIBRARY})
  else()
    message(
      FATAL_ERROR
        "Velox temp_path archive is missing, but the source target is not header-only: ${VELOX_TEMP_PATH_LIBRARY}"
    )
  endif()
  import_library(
    facebook::velox::exec_test_lib
    ${VELOX_BUILD_PATH}/velox/exec/tests/utils/libvelox_exec_test_lib.a)
  target_link_libraries(
    facebook::velox::exec_test_lib
    INTERFACE facebook::velox::vector_test_lib
              facebook::velox::dwio_common_test
              facebook::velox::file_test_utils
              facebook::velox::temp_path
              # QueryAssertions and OperatorTestBase in exec_test_lib call
              # DuckDB directly. Keep DuckDB after the importing archive on
              # one-pass GNU linker command lines.
              duckdb_static)
  target_link_libraries(velox PUBLIC facebook::velox::exec_test_lib)
endif()

# PRIVATE + whole-archive: facebook::velox is libvelox.a (STATIC IMPORTED).
# Baked into libgluten.so; must NOT propagate to consumers or its globals
# (gflags like velox_memory_num_shared_leaf_pools) fire twice. `--whole-archive`
# pulls in every .o so that APIs consumers call but libgluten.so's own code
# doesn't reference (e.g. folly::Init, various init helpers) are still present
# in libgluten.so for dynamic symbol lookup.
target_link_libraries(
  velox PRIVATE -Wl,--whole-archive facebook::velox -Wl,--no-whole-archive
                facebook::velox::dbgen)

# PRIVATE + whole-archive: Folly::folly is libfolly.a (STATIC IMPORTED). Same
# reasoning as facebook::velox above — PRIVATE prevents double-registration of
# folly's own gflags in consumers; whole-archive ensures consumer-only APIs
# (folly::Init, etc.) are present in libgluten.so for symbol lookup.
target_link_libraries(velox PRIVATE -Wl,--whole-archive Folly::folly
                                    -Wl,--no-whole-archive)

# The velox target embeds the static gluten core archive manually with
# --whole-archive, so close gluten's public static dependencies on the final
# shared library instead of relying on transitive propagation through that
# linker-flag hunk.
find_arrow_lib(${ARROW_LIB_NAME})
find_arrow_lib(${ARROW_BUNDLED_DEPS})
target_link_libraries(velox PRIVATE Arrow::arrow
                                    Arrow::arrow_bundled_dependencies)

# glog is SHARED (libglog.so); consumers whose code calls google::LOG /
# google::SetLogDestination need it on their link line. Propagate PUBLIC so
# downstream executables/plugins inherit it automatically. (The PRIVATE chain
# through `gluten -> google::glog` used to cover this implicitly; no longer
# propagates now that the gluten whole-archive hunk is PRIVATE.)
target_link_libraries(velox PUBLIC google::glog)

find_re2()
target_link_libraries(velox PUBLIC ${RE2_LIBRARY})

set(CMAKE_FIND_LIBRARY_SUFFIXES_BCK ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_library(STEMMER_LIB NAMES libstemmer.a)
if(STEMMER_LIB)
  message(STATUS "Found stemmer: ${STEMMER_LIB}")
  add_library(external::stemmer STATIC IMPORTED)
  set_target_properties(external::stemmer PROPERTIES IMPORTED_LOCATION
                                                     ${STEMMER_LIB})
else()
  import_library(
    external::stemmer
    ${VELOX_BUILD_PATH}/_deps/libstemmer/src/libstemmer/libstemmer.a)
endif()
target_link_libraries(velox PUBLIC external::stemmer)

find_package(geos)
if(geos_FOUND AND TARGET GEOS::geos)
  add_library(external::geos ALIAS GEOS::geos)
else()
  message(STATUS "import Velox bundled geos")
  import_library(external::geos ${VELOX_BUILD_PATH}/lib/libgeos.a)
endif()
target_link_libraries(velox PUBLIC external::geos)

find_package(simdjson CONFIG)
if(simdjson_FOUND AND TARGET simdjson::simdjson)
  target_link_libraries(velox PUBLIC simdjson::simdjson)
else()
  import_library(external::simdjson
                 ${VELOX_BUILD_PATH}/_deps/simdjson-build/libsimdjson.a)
  target_link_libraries(velox PUBLIC external::simdjson)
endif()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_BCK})

# Adopted from Velox's CMakeLists.txt.
# https://github.com/facebookincubator/velox/pull/11410
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  execute_process(
    COMMAND brew --prefix icu4c
    RESULT_VARIABLE BREW_ICU4C
    OUTPUT_VARIABLE BREW_ICU4C_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(BREW_ICU4C EQUAL 0 AND EXISTS "${BREW_ICU4C_PREFIX}")
    message(STATUS "Found icu4c installed by Homebrew at ${BREW_ICU4C_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${BREW_ICU4C_PREFIX}")
  else()
    list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/icu4c")
  endif()
endif()

find_package(
  ICU
  COMPONENTS i18n uc data
  REQUIRED)
target_link_libraries(velox PUBLIC ICU::i18n ICU::uc ICU::data)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(ENABLE_HDFS)
  add_definitions(-DENABLE_HDFS)
endif()

if(ENABLE_S3)
  add_definitions(-DENABLE_S3)
  find_awssdk()
  target_link_libraries(velox PUBLIC ${AWSSDK_LIBRARIES})
endif()

if(ENABLE_GCS)
  add_definitions(-DENABLE_GCS)
  find_gcssdk()
  target_link_libraries(velox PUBLIC google-cloud-cpp::storage)
endif()

if(ENABLE_ABFS)
  add_definitions(-DENABLE_ABFS)
  find_azure()
  target_link_libraries(velox PUBLIC Azure::azure-storage-blobs)
  target_link_libraries(velox PUBLIC Azure::azure-storage-files-datalake)
  target_link_libraries(velox PUBLIC Azure::azure-identity)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(udf/examples)
endif()

# Standalone developer tool binaries (flux-substrait-runner, etc). Always added
# so `cmake --build --target flux-substrait-runner` works without needing to
# re-run cmake with an extra flag. The binaries link against the existing
# `velox` target so they incur no extra heavy deps.
add_subdirectory(tools)

if(ENABLE_GPU)
  # Propagate GPU flag to velox target so FluxJniWrapper.cc and other sources
  # can use #ifdef GLUTEN_ENABLE_GPU for GPU-specific code paths.
  target_compile_definitions(velox PRIVATE GLUTEN_ENABLE_GPU)

  # A cuDF build-tree package config resolves transitive RAPIDS dependencies
  # through CPMFindPackage. Standalone Gluten reconfiguration does not load
  # Velox's top-level CPM bootstrap, so load the already-generated CPM module
  # for BUNDLED builds. An installed SYSTEM package resolves its dependencies
  # through its exported package closure and must not depend on that build tree.
  if(cudf_SOURCE STREQUAL "BUNDLED" AND NOT COMMAND CPMFindPackage)
    # CPM records initialization in a GLOBAL property, but CMake command
    # definitions are directory-scoped. Another subproject can therefore set the
    # property without making CPMFindPackage visible in this directory. Clear
    # the stale marker before including CPM in the Velox backend scope.
    set_property(GLOBAL PROPERTY CPM_INITIALIZED)
    if(DEFINED CPM_FILE AND EXISTS "${CPM_FILE}")
      get_filename_component(CPM_FILE "${CPM_FILE}" REALPATH)
      get_filename_component(CPM_DIRECTORY "${CPM_FILE}" DIRECTORY)
      include("${CPM_FILE}")
    elseif(EXISTS "${VELOX_BUILD_PATH}/cmake/CPM_0.42.0.cmake")
      set(CPM_FILE "${VELOX_BUILD_PATH}/cmake/CPM_0.42.0.cmake")
      get_filename_component(CPM_FILE "${CPM_FILE}" REALPATH)
      get_filename_component(CPM_DIRECTORY "${CPM_FILE}" DIRECTORY)
      include("${CPM_FILE}")
    else()
      message(
        FATAL_ERROR "CPMFindPackage is required to import the cuDF build tree")
    endif()
  endif()

  find_package(cudf CONFIG REQUIRED)
  if(cudf_SOURCE STREQUAL "SYSTEM")
    find_package(ucxx CONFIG REQUIRED)
  endif()

  import_library(
    facebook::velox::velox_cudf_expression
    ${VELOX_BUILD_PATH}/velox/experimental/cudf/expression/libvelox_cudf_expression.a
  )
  import_library(
    facebook::velox::velox_cudf_exec
    ${VELOX_BUILD_PATH}/velox/experimental/cudf/exec/libvelox_cudf_exec.a)
  import_library(
    facebook::velox::velox_cudf_vector
    ${VELOX_BUILD_PATH}/velox/experimental/cudf/vector/libvelox_cudf_vector.a)
  import_library(
    facebook::velox::velox_cudf_hive_connector
    ${VELOX_BUILD_PATH}/velox/experimental/cudf/connectors/hive/libvelox_cudf_hive_connector.a
  )
  import_library(
    facebook::velox::velox_cudf_iceberg_connector
    ${VELOX_BUILD_PATH}/velox/experimental/cudf/connectors/hive/iceberg/libvelox_cudf_iceberg_connector.a
  )
  # IBM-baseline velox dropped velox/experimental/cudf/exchange/. Its
  # replacement velox_ucx_exchange object library is already embedded in
  # velox_cudf_exec and must be linked exactly once. Linking the standalone
  # objects again registers gflags and process-wide UCX singletons twice.

  # The combined Gluten library needs both Apache Thrift (used by the Spark
  # RAPIDS JNI dependency) and the current FBThrift static archives used by
  # Velox.  Older build images happened to install all of them under
  # /usr/local/lib; discover them explicitly so a separately-built Velox tree
  # can be used as well.
  set(GLUTEN_APACHE_THRIFT_LIBRARY
      ""
      CACHE FILEPATH "Apache Thrift library used by the cuDF JNI dependency")
  if(NOT GLUTEN_APACHE_THRIFT_LIBRARY)
    unset(GLUTEN_APACHE_THRIFT_LIBRARY CACHE)
    find_library(
      GLUTEN_APACHE_THRIFT_LIBRARY
      NAMES thrift
      PATHS /usr/local/lib /usr/local/lib64)
  endif()
  if(NOT GLUTEN_APACHE_THRIFT_LIBRARY)
    message(FATAL_ERROR "Apache Thrift library (libthrift.a) not found")
  endif()

  set(GLUTEN_FBTHRIFT_LIBRARY_DIR
      ""
      CACHE PATH "Directory containing the FBThrift static archives")
  if(NOT GLUTEN_FBTHRIFT_LIBRARY_DIR)
    unset(GLUTEN_FBTHRIFT_LIBRARY_DIR CACHE)
    find_path(
      GLUTEN_FBTHRIFT_LIBRARY_DIR
      NAMES libthriftcpp2.a
      PATHS /usr/local/lib /usr/local/lib64
            ${VELOX_BUILD_PATH}/_deps/fbthrift-build/lib)
  endif()
  if(NOT GLUTEN_FBTHRIFT_LIBRARY_DIR)
    message(FATAL_ERROR "FBThrift static library directory not found")
  endif()
  set(GLUTEN_FBTHRIFT_LIBRARIES)
  foreach(
    _gluten_fbthrift_name IN
    ITEMS thriftcpp2
          thriftprotocol
          thrift-core
          thrifttyperep
          thriftannotation
          thrifttype
          thriftanyrep
          thriftmetadata
          thriftfrozen2)
    set(_gluten_fbthrift_path
        "${GLUTEN_FBTHRIFT_LIBRARY_DIR}/lib${_gluten_fbthrift_name}.a")
    if(NOT EXISTS "${_gluten_fbthrift_path}")
      message(FATAL_ERROR "Missing FBThrift archive: ${_gluten_fbthrift_path}")
    endif()
    list(APPEND GLUTEN_FBTHRIFT_LIBRARIES "${_gluten_fbthrift_path}")
  endforeach()

  target_compile_definitions(
    velox PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

  add_library(velox_curl SHARED IMPORTED GLOBAL)
  set_target_properties(
    velox_curl PROPERTIES IMPORTED_LOCATION
                          ${VELOX_BUILD_PATH}/_deps/curl-build/lib/libcurl.so)

  # Static cuDF integration archives: PRIVATE + whole-archive. Like
  # facebook::velox, these need every .o pulled in so consumer-used APIs end up
  # exported from libgluten.so. Use full paths instead of imported targets so
  # cmake doesn't drop the --whole-archive flags.
  target_link_libraries(
    velox
    PRIVATE
      -Wl,--whole-archive
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/exec/libvelox_cudf_exec.a
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/vector/libvelox_cudf_vector.a
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/connectors/hive/libvelox_cudf_hive_connector.a
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/connectors/hive/iceberg/libvelox_cudf_iceberg_connector.a
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/expression/libvelox_cudf_expression.a
      ${VELOX_BUILD_PATH}/velox/experimental/cudf/expression/libvelox_cudf_expression_registry.a
      -Wl,--no-whole-archive)
  # The cuDF connector archives above contain S3 callers. Repeat the static AWS
  # SDK closure after those archives so one-pass GNU linking retains the
  # corresponding S3/S3 CRT objects in libgluten.so itself.
  if(ENABLE_S3)
    target_link_libraries(velox PRIVATE ${AWSSDK_LIBRARIES})
  endif()
  # Shared libs consumers need at link time (cudf::cudf, libcurl): PUBLIC.
  target_link_libraries(velox PUBLIC cudf::cudf velox_curl)
  target_link_libraries(velox PUBLIC ${GLUTEN_APACHE_THRIFT_LIBRARY})
  # Velox's command-line/configuration code uses Boost.Program_options.  The
  # static Velox archive leaves those symbols unresolved, so make the shared
  # dependency explicit on libgluten.so instead of relying on the host JVM
  # process to preload it.
  find_library(
    GLUTEN_BOOST_PROGRAM_OPTIONS_LIBRARY
    NAMES boost_program_options
    PATHS /usr/local/lib64 /usr/local/lib)
  if(NOT GLUTEN_BOOST_PROGRAM_OPTIONS_LIBRARY)
    message(FATAL_ERROR "Boost.Program_options library not found")
  endif()
  target_link_libraries(velox PUBLIC ${GLUTEN_BOOST_PROGRAM_OPTIONS_LIBRARY})
  foreach(_gluten_boost_component IN ITEMS filesystem system context)
    string(TOUPPER "${_gluten_boost_component}" _gluten_boost_component_upper)
    find_library(
      GLUTEN_BOOST_${_gluten_boost_component_upper}_LIBRARY
      NAMES boost_${_gluten_boost_component}
      PATHS /usr/local/lib64 /usr/local/lib)
    if(NOT GLUTEN_BOOST_${_gluten_boost_component_upper}_LIBRARY)
      message(FATAL_ERROR "Boost.${_gluten_boost_component} library not found")
    endif()
    target_link_libraries(
      velox PUBLIC ${GLUTEN_BOOST_${_gluten_boost_component_upper}_LIBRARY})
  endforeach()
  foreach(_gluten_runtime_name IN ITEMS zstd event_core sodium)
    find_library(GLUTEN_${_gluten_runtime_name}_LIBRARY
                 NAMES ${_gluten_runtime_name})
    if(NOT GLUTEN_${_gluten_runtime_name}_LIBRARY)
      message(FATAL_ERROR "Runtime library not found: ${_gluten_runtime_name}")
    endif()
    target_link_libraries(velox
                          PUBLIC ${GLUTEN_${_gluten_runtime_name}_LIBRARY})
  endforeach()
  find_library(
    GLUTEN_FMT_LIBRARY
    NAMES fmt
    PATHS ${VELOX_BUILD_PATH}/_deps/fmt-build)
  if(NOT GLUTEN_FMT_LIBRARY)
    message(FATAL_ERROR "Bundled fmt archive not found")
  endif()
  target_link_libraries(velox PUBLIC ${GLUTEN_FMT_LIBRARY})
  find_library(GLUTEN_DOUBLE_CONVERSION_LIBRARY NAMES double-conversion)
  if(NOT GLUTEN_DOUBLE_CONVERSION_LIBRARY)
    message(FATAL_ERROR "double-conversion archive not found")
  endif()
  target_link_libraries(velox PUBLIC ${GLUTEN_DOUBLE_CONVERSION_LIBRARY})
  find_library(GLUTEN_SNAPPY_LIBRARY NAMES snappy)
  if(NOT GLUTEN_SNAPPY_LIBRARY)
    message(FATAL_ERROR "Snappy library not found")
  endif()
  target_link_libraries(velox PUBLIC ${GLUTEN_SNAPPY_LIBRARY})
  # Legacy RAPIDS layouts expose these dependencies as shared libraries and
  # require explicit links. Current cuDF builds absorb their static archives
  # into libcudf.so; their build-tree locations may also disappear when CMake
  # reuses installed packages. Validate the shared cuDF artifact once, retain
  # legacy shared links when present, and never link a second static copy.
  get_target_property(_gluten_cudf_type cudf::cudf TYPE)
  get_target_property(_gluten_cudf_location cudf::cudf
                      IMPORTED_LOCATION_RELEASE)
  if(NOT _gluten_cudf_type STREQUAL "SHARED_LIBRARY"
     OR NOT EXISTS "${_gluten_cudf_location}")
    message(
      FATAL_ERROR
        "RAPIDS dependencies require an existing shared cudf::cudf; "
        "observed type=${_gluten_cudf_type}, "
        "location=${_gluten_cudf_location}")
  endif()

  set(GLUTEN_EXPLICIT_RAPIDS_LIBRARIES)
  if(cudf_SOURCE STREQUAL "SYSTEM")
    set(GLUTEN_UCXX_LIBRARY ucxx::ucxx)
  else()
    set(_gluten_rmm_shared "${VELOX_BUILD_PATH}/_deps/rmm-build/librmm.so")
    if(EXISTS "${_gluten_rmm_shared}")
      list(APPEND GLUTEN_EXPLICIT_RAPIDS_LIBRARIES "${_gluten_rmm_shared}")
      message(STATUS "Using legacy shared RMM library: ${_gluten_rmm_shared}")
    else()
      message(
        STATUS "Using RMM absorbed by shared cuDF: ${_gluten_cudf_location}")
    endif()

    set(_gluten_logger_shared
        "${VELOX_BUILD_PATH}/_deps/rapids_logger-build/librapids_logger.so")
    if(EXISTS "${_gluten_logger_shared}")
      list(APPEND GLUTEN_EXPLICIT_RAPIDS_LIBRARIES "${_gluten_logger_shared}")
      message(
        STATUS
          "Using legacy shared RAPIDS logger library: ${_gluten_logger_shared}")
    else()
      message(
        STATUS
          "Using RAPIDS logger absorbed by shared cuDF: ${_gluten_cudf_location}"
      )
    endif()

    set(_gluten_kvikio_shared
        "${VELOX_BUILD_PATH}/_deps/kvikio-build/libkvikio.so")
    if(EXISTS "${_gluten_kvikio_shared}")
      list(APPEND GLUTEN_EXPLICIT_RAPIDS_LIBRARIES "${_gluten_kvikio_shared}")
      message(
        STATUS "Using legacy shared KvikIO library: ${_gluten_kvikio_shared}")
    else()
      find_library(
        _gluten_kvikio_system
        NAMES kvikio
        PATHS /usr/local/lib64 /usr/local/lib
        NO_DEFAULT_PATH)
      if(_gluten_kvikio_system)
        list(APPEND GLUTEN_EXPLICIT_RAPIDS_LIBRARIES "${_gluten_kvikio_system}")
        message(STATUS "Using system shared KvikIO library: "
                       "${_gluten_kvikio_system}")
      else()
        message(
          STATUS
            "Using KvikIO absorbed by shared cuDF: ${_gluten_cudf_location}")
      endif()
    endif()

    set(_gluten_ucxx_bundled "${VELOX_BUILD_PATH}/_deps/ucxx-build/libucxx.so")
    if(EXISTS "${_gluten_ucxx_bundled}")
      set(GLUTEN_UCXX_LIBRARY "${_gluten_ucxx_bundled}")
    else()
      find_library(
        _gluten_ucxx_system
        NAMES ucxx
        PATHS /usr/local/lib64 /usr/local/lib
        NO_DEFAULT_PATH)
      set(GLUTEN_UCXX_LIBRARY "${_gluten_ucxx_system}")
    endif()
    if(NOT GLUTEN_UCXX_LIBRARY OR NOT EXISTS "${GLUTEN_UCXX_LIBRARY}")
      message(
        FATAL_ERROR
          "GPU builds require either Velox's bundled UCXX artifact or an "
          "installed libucxx.so")
    endif()
  endif()

  set(GLUTEN_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)
  if(EXISTS /usr/local/cuda/lib64/libcudart.so.13)
    set(GLUTEN_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so.13)
  endif()
  message(STATUS "Using UCXX library: ${GLUTEN_UCXX_LIBRARY}")

  target_link_libraries(
    velox PUBLIC ${GLUTEN_EXPLICIT_RAPIDS_LIBRARIES} ${GLUTEN_UCXX_LIBRARY}
                 ${GLUTEN_CUDART_LIBRARY} ${GLUTEN_UCX_LIBRARIES})
endif()

# Keep these at the end of libgluten.so's link line. `facebook::velox` and the
# cuDF integration archives are pulled in with whole-archive above, and static
# dependencies must appear after the archives that reference them.
target_link_libraries(
  velox
  PRIVATE duckdb_static
          -Wl,--start-group
          Arrow::arrow
          Arrow::arrow_bundled_dependencies
          facebook::velox::dwio_parquet_thrift_raw
          external::s2
          ${GLUTEN_APACHE_THRIFT_LIBRARY}
          ${GLUTEN_FBTHRIFT_LIBRARIES}
          ${GLUTEN_ABSL_LIBRARIES}
          ${GLUTEN_QUADMATH_LIBRARY}
          -Wl,--end-group)

add_custom_command(
  TARGET velox
  POST_BUILD
  COMMAND ld $<TARGET_FILE:velox> || true
  COMMENT "Checking ld result of libgluten.so (combined)")
add_custom_command(
  TARGET velox
  POST_BUILD
  COMMAND ldd $<TARGET_FILE:velox> || true
  COMMENT "Checking ldd result of libgluten.so (combined)")
