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

# Temporary premerge repair layer for prebuild images that contain the CCCL
# CMake metadata but not its headers. CCCL is header-only, so this does not
# rebuild cuDF.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# CCCL 3.4.0 matches the version selected by the cuDF 26.06 dependency stack.
ARG CCCL_COMMIT=a5630ac57111e1aaef8cfe0069ede566c3d15f60
ARG CCCL_SHA256=cbd4051441fe2fa19fe0da2987e8dea90bc51fe9d2111e55b75b0ec1301f33ae

RUN source /opt/rh/gcc-toolset-14/enable \
  && set -eux \
  && curl --retry 5 --retry-all-errors --retry-delay 2 -fL \
       "https://github.com/NVIDIA/cccl/archive/${CCCL_COMMIT}.tar.gz" \
       -o /tmp/cccl.tar.gz \
  && echo "${CCCL_SHA256}  /tmp/cccl.tar.gz" | sha256sum -c - \
  && mkdir -p /tmp/cccl-src \
  && tar -xzf /tmp/cccl.tar.gz -C /tmp/cccl-src --strip-components=1 \
  && cmake -S /tmp/cccl-src -B /tmp/cccl-install -GNinja \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCCCL_ENABLE_TESTING=OFF \
       -DCCCL_ENABLE_EXAMPLES=OFF \
       -DCCCL_ENABLE_BENCHMARKS=OFF \
  && cmake --install /tmp/cccl-install \
  && test -f /usr/local/include/cuda/version \
  && printf '#include <cuda/version>\n' | g++ -x c++ -E - >/dev/null \
  && rm -rf /tmp/cccl.tar.gz /tmp/cccl-src /tmp/cccl-install
