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

# mpp-substrait-runner: standalone binary that replays Gluten-dumped
# multi-fragment Substrait plans on Velox-cudf via MppQueryCoordinator.
#
# Links against the `velox` shared library (libgluten.so) which already
# contains MppQueryCoordinator, SubstraitToVeloxPlan, and (when ENABLE_GPU)
# the cuDF exchange components.

add_executable(mpp-substrait-runner MppSubstraitRunner.cc)

target_include_directories(
  mpp-substrait-runner
  PRIVATE ${CMAKE_SOURCE_DIR}/velox
          ${CMAKE_SOURCE_DIR}/core
          ${VELOX_HOME}
          ${VELOX_BUILD_PATH})

# Link against the velox backend shared library (produces libgluten.so).
# That library already pulls in facebook::velox and, when ENABLE_GPU is on,
# the cuDF exchange / hive-connector static archives.
target_link_libraries(mpp-substrait-runner PRIVATE velox)

# fmt is a pure template/helper library with no global registration state.
# Some of its internals (e.g. fmt::v11::detail::is_printable) are compiled
# with LOCAL visibility inside libfmt.a and therefore cannot be exported
# from libgluten.so via the version script. Link libfmt.a directly here —
# double-linking it is safe because it has no gflag / protobuf / init
# registration that would fire twice.
find_package(fmt CONFIG QUIET)
if(TARGET fmt::fmt)
  target_link_libraries(mpp-substrait-runner PRIVATE fmt::fmt)
else()
  find_library(FMT_LIBRARY fmt REQUIRED)
  target_link_libraries(mpp-substrait-runner PRIVATE ${FMT_LIBRARY})
endif()

if(ENABLE_GPU)
  target_compile_definitions(mpp-substrait-runner PRIVATE GLUTEN_ENABLE_GPU)
  target_link_libraries(mpp-substrait-runner PRIVATE velox_curl)
endif()

# If Folly::folly resolves to the shared libfolly.so (which is what the
# stock container layout produces), libgluten.so won't have folly baked in
# and the executable's direct folly::Init use triggers a "DSO missing from
# command line" link error. Find and link libfolly.so explicitly so the
# executable carries its own DT_NEEDED.
find_package(Folly QUIET CONFIG)
if(TARGET Folly::folly)
  target_link_libraries(mpp-substrait-runner PRIVATE Folly::folly)
endif()

# Helpful install location so the binary lands next to the other artifacts.
set_target_properties(
  mpp-substrait-runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                  ${root_directory}/releases)
