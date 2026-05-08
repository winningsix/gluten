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
  find_package(AWSSDK REQUIRED COMPONENTS s3;identity-management)
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
    compute/MppDumpLoader.cc
    compute/MppQueryCoordinator.cc
    compute/VeloxBackend.cc
    compute/VeloxRuntime.cc
    compute/VeloxPlanConverter.cc
    compute/WholeStageResultIterator.cc
    compute/iceberg/IcebergPlanConverter.cc
    jni/JniFileSystem.cc
    jni/JniUdf.cc
    jni/MppJniWrapper.cc
    jni/VeloxJniWrapper.cc
    memory/BufferOutputStream.cc
    memory/VeloxColumnarBatch.cc
    memory/VeloxMemoryManager.cc
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

if(ENABLE_S3)
  find_package(ZLIB)
endif()

if(ENABLE_GPU)
  list(
    APPEND
    VELOX_SRCS
    cudf/CudfPlanValidator.cc
    cudf/GpuLock.cc
    cudf/GpuMemoryTracker.cc
    operators/plannodes/CudfVectorStream.cc
    shuffle/VeloxGpuShuffleReader.cc
    shuffle/VeloxGpuShuffleWriter.cc
    operators/serializer/VeloxGpuColumnarBatchSerializer.cc
    utils/GpuBufferBatchResizer.cc
    memory/GpuBufferColumnarBatch.cc)
endif()

if(ENABLE_ENHANCED_FEATURES)
  list(APPEND VELOX_SRCS compute/iceberg/IcebergFormat.cc
       compute/iceberg/IcebergWriter.cc)
endif()

if(BUILD_TESTS OR BUILD_BENCHMARKS)
  set(BUILD_TEST_UTILS ON)
endif()

if(BUILD_TEST_UTILS)
  list(APPEND VELOX_SRCS tests/utils/LocalRssClient.cc
       tests/utils/TestAllocationListener.cc)
endif()

add_library(velox SHARED ${VELOX_SRCS})

# Produce libgluten.so instead of libvelox.so so that a single shared library
# contains both core Gluten JNI symbols and backend (Velox/MPP) JNI symbols.
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

target_include_directories(
  velox
  PUBLIC ${CMAKE_SYSTEM_INCLUDE_PATH}
         ${JNI_INCLUDE_DIRS}
         ${VELOX_BUILD_PATH}
         # ucxx headers fetched by velox FetchContent. The release build
         # tree (VELOX_BUILD_PATH) does not stage them under _deps, so
         # point at the ucx-probe build dir explicitly. Required by
         # velox/experimental/ucx-exchange/Communicator.h's
         # `#include "ucxx/api.h"`.
         /opt/velox/_build/ucx-probe/_deps/ucxx-src/cpp/include
         ${VELOX_BUILD_PATH}/_deps/ucxx-src/cpp/include
         # cudf + transitive dep headers fetched by velox FetchContent.
         # cudf interop types (e.g. cudf/io/types.hpp) aren't propagated
         # through velox cudf static-library targets; cudf headers
         # transitively include rmm + cccl + others, so we list each
         # FetchContent _src/include subtree explicitly.
         ${VELOX_BUILD_PATH}/_deps/cudf-src/cpp/include
         ${VELOX_BUILD_PATH}/_deps/rmm-src/cpp/include
         ${VELOX_BUILD_PATH}/_deps/rmm-build/include
         ${VELOX_BUILD_PATH}/_deps/cccl-src/libcudacxx/include
         ${VELOX_BUILD_PATH}/_deps/cccl-src/cub
         ${VELOX_BUILD_PATH}/_deps/cccl-src/thrust
         ${VELOX_BUILD_PATH}/_deps/cuco-src/include
         ${VELOX_BUILD_PATH}/_deps/kvikio-src/cpp/include
         ${VELOX_BUILD_PATH}/_deps/rapids_logger-src/include
         # CUDA runtime headers (cuda_runtime_api.h etc.) used transitively
         # by rmm/cuda_stream_view.hpp.
         /usr/local/cuda/include
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${VELOX_HOME}
         ${PROTO_OUTPUT_DIR}
         ${PROTOBUF_INCLUDE}
         # Forward gluten's public include dirs (substrait/*.pb.h etc.) to
         # downstream consumers. Previously propagated transitively via
         # `target_link_libraries(velox PUBLIC ... gluten ...)`, which now
         # only lives in velox's PRIVATE scope to avoid baking libgluten.a
         # into consumers a second time.
         $<TARGET_PROPERTY:gluten,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(velox PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                       ${root_directory}/releases)

# If folly is not installed in system lib paths, please add
# `-DCMAKE_PREFIX_PATH="${folly lib path}" to cmake arguments. It is also
# applicable to other dependencies.
find_package(Folly REQUIRED CONFIG)

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
# points (referenced only from Java, not from C++) are not dropped by the linker.
# Keep this PRIVATE: the whole-archive/allow-multiple-definition dance is only
# needed for libgluten.so's own build. Propagating it via PUBLIC causes every
# downstream consumer (executables, plugin .so's) to bake libgluten.a +
# libprotobuf.a into themselves a second time, which FATALs at runtime when
# protobuf's global descriptor pool detects the same .proto registered twice.
target_link_libraries(velox PRIVATE
  -Wl,--whole-archive gluten -Wl,--no-whole-archive
  -Wl,--allow-multiple-definition)

# Requires VELOX_MONO_LIBRARY=ON when building Velox.
import_library(facebook::velox ${VELOX_BUILD_PATH}/lib/libvelox.a)

# velox/tpch/gen generator functions (mk_region, sd_cust, etc.) live in
# libdbgen.a. When libvelox.a is whole-archive'd into libgluten.so, velox's
# tpch code is pulled in and needs these symbols regardless of BUILD_TESTS.
import_library(facebook::velox::dbgen
               ${VELOX_BUILD_PATH}/velox/tpch/gen/dbgen/libdbgen.a)
add_duckdb()

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
  import_library(
    facebook::velox::temp_path
    ${VELOX_BUILD_PATH}/velox/exec/tests/utils/libvelox_temp_path.a)
  import_library(
    facebook::velox::exec_test_lib
    ${VELOX_BUILD_PATH}/velox/exec/tests/utils/libvelox_exec_test_lib.a)
  target_link_libraries(
    facebook::velox::exec_test_lib
    INTERFACE facebook::velox::vector_test_lib
              facebook::velox::dwio_common_test
              facebook::velox::file_test_utils facebook::velox::temp_path)
  target_link_libraries(velox PUBLIC facebook::velox::exec_test_lib)
endif()

# PRIVATE + whole-archive: facebook::velox is libvelox.a (STATIC IMPORTED).
# Baked into libgluten.so; must NOT propagate to consumers or its globals
# (gflags like velox_memory_num_shared_leaf_pools) fire twice. `--whole-archive`
# pulls in every .o so that APIs consumers call but libgluten.so's own code
# doesn't reference (e.g. folly::Init, various init helpers) are still present
# in libgluten.so for dynamic symbol lookup.
target_link_libraries(velox PRIVATE
  -Wl,--whole-archive facebook::velox -Wl,--no-whole-archive
  facebook::velox::dbgen)

# PRIVATE + whole-archive: Folly::folly is libfolly.a (STATIC IMPORTED).
# Same reasoning as facebook::velox above — PRIVATE prevents double-registration
# of folly's own gflags in consumers; whole-archive ensures consumer-only APIs
# (folly::Init, etc.) are present in libgluten.so for symbol lookup.
target_link_libraries(velox PRIVATE
  -Wl,--whole-archive Folly::folly -Wl,--no-whole-archive)

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

# Standalone developer tool binaries (mpp-substrait-runner, etc).
# Always added so `cmake --build --target mpp-substrait-runner` works
# without needing to re-run cmake with an extra flag. The binaries link
# against the existing `velox` target so they incur no extra heavy deps.
add_subdirectory(tools)

if(ENABLE_GPU)
  # Propagate GPU flag to velox target so MppJniWrapper.cc and other sources
  # can use #ifdef GLUTEN_ENABLE_GPU for GPU-specific code paths.
  target_compile_definitions(velox PRIVATE GLUTEN_ENABLE_GPU)

  find_package(cudf CONFIG REQUIRED)

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
  # IBM-baseline velox dropped velox/experimental/cudf/exchange/. The
  # equivalent functionality is in IBM's UcxExchange/UcxPartitionedOutput
  # which live under velox/experimental/ucx-exchange/, linked transitively
  # via velox_cudf_exec.

  target_compile_definitions(
    velox PRIVATE LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

  add_library(velox_curl SHARED IMPORTED GLOBAL)
  set_target_properties(
    velox_curl PROPERTIES IMPORTED_LOCATION
                          ${VELOX_BUILD_PATH}/_deps/curl-build/lib/libcurl.so)

  # Static cuDF integration archives: PRIVATE + whole-archive. Like
  # facebook::velox, these need every .o pulled in so consumer-used APIs
  # end up exported from libgluten.so. Use full paths instead of imported
  # targets so cmake doesn't drop the --whole-archive flags.
  target_link_libraries(
    velox
    PRIVATE -Wl,--whole-archive
            ${VELOX_BUILD_PATH}/velox/experimental/cudf/exec/libvelox_cudf_exec.a
            ${VELOX_BUILD_PATH}/velox/experimental/cudf/vector/libvelox_cudf_vector.a
            ${VELOX_BUILD_PATH}/velox/experimental/cudf/connectors/hive/libvelox_cudf_hive_connector.a
            ${VELOX_BUILD_PATH}/velox/experimental/cudf/expression/libvelox_cudf_expression.a
            -Wl,--no-whole-archive)
  # Shared libs consumers need at link time (cudf::cudf, libcurl): PUBLIC.
  target_link_libraries(velox PUBLIC cudf::cudf velox_curl)
  # IBM baseline brings in apache thrift symbols transitively (Presto serde
  # uses thrift for handshake / buffered transport). System Arrow doesn't
  # link them in, so add /usr/local/lib/libthrift.a here.
  target_link_libraries(velox PUBLIC /usr/local/lib/libthrift.a)
  # rmm cuda_stream_view + cudart_static + ucxx need explicit shared lib
  # links since the static archives forward-declare but don't link them.
  target_link_libraries(velox PUBLIC
    ${VELOX_BUILD_PATH}/_deps/rmm-build/librmm.so
    ${VELOX_BUILD_PATH}/_deps/rapids_logger-build/librapids_logger.so
    ${VELOX_BUILD_PATH}/_deps/kvikio-build/libkvikio.so
    ${VELOX_BUILD_PATH}/_deps/ucxx-build/libucxx.so
    /usr/local/cuda/lib64/libcudart.so
    /usr/lib64/libucp.so
    /usr/lib64/libuct.so
    /usr/lib64/libucs.so)
endif()

# Keep these at the end of libgluten.so's link line. `facebook::velox` and the
# cuDF integration archives are pulled in with whole-archive above, and static
# dependencies must appear after the archives that reference them.
target_link_libraries(velox PRIVATE
  duckdb_static
  -Wl,--start-group Arrow::arrow Arrow::arrow_bundled_dependencies -Wl,--end-group)

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
