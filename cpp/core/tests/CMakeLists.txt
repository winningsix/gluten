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

add_test_case(round_robin_partitioner_test SOURCES RoundRobinPartitionerTest.cc)
add_test_case(object_store_test SOURCES ObjectStoreTest.cc)
add_test_case(parallel_compression_test SOURCES ParallelCompressionTest.cc)

add_executable(parallel_compression_bench ParallelCompressionBench.cc)
target_include_directories(parallel_compression_bench PRIVATE ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(parallel_compression_bench gluten Arrow::arrow)
