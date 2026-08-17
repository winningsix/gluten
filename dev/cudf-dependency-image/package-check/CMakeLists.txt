# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements.  See the NOTICE file distributed with this work for
# additional information regarding copyright ownership. The ASF licenses this
# file to You under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 4.0)
project(gluten_cudf_dependency_check LANGUAGES CXX)

find_package(cudf CONFIG REQUIRED)
find_package(ucxx CONFIG REQUIRED)
find_package(nvtx3 CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CURL CONFIG REQUIRED)
find_package(AWSSDK CONFIG REQUIRED COMPONENTS s3 identity-management s3-crt)

foreach(target IN ITEMS cudf::cudf ucxx::ucxx nvtx3::nvtx3-cpp)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Required installed target is missing: ${target}")
  endif()
endforeach()

if(NOT AWSSDK_LIBRARIES)
  message(FATAL_ERROR "AWSSDK did not expose the requested component closure")
endif()

# Compile and link a real consumer so missing public headers, S3/S3-CRT and
# identity-management archives, or transitive AWS CRT libraries fail carrier
# admission. The checker builds but does not run this executable, so no
# credentials or network access are needed.
add_executable(aws_s3_link_probe aws-s3-link-probe.cpp)
target_compile_features(aws_s3_link_probe PRIVATE cxx_std_17)
target_link_libraries(aws_s3_link_probe PRIVATE ${AWSSDK_LIBRARIES})
