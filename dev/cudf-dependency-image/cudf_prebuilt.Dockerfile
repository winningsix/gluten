# syntax=docker/dockerfile:1.7
#
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

# Purpose:
# Build a reusable dependency-only carrier for downstream Spark-Gluten builds
# that select the existing SYSTEM-cuDF path. The image supplies the compiler
# toolchain with JDK 17, installed cuDF/RAPIDS dependencies, Spark-Gluten's
# patched Arrow Java artifacts, and the selected Velox tree's pinned static AWS
# S3/S3-CRT dependency closure. It never carries compiled Velox or Gluten
# consumers.
#
# Image architecture:
# - `toolchain` aligns the inherited GCC/CMake environment, installs CMake,
#   Maven, JDK 17, and patchelf, removes the base image's partial Arrow
#   installation, installs a current curl, and builds the checksum-pinned Velox
#   UCX release with CUDA enabled. UCX core libraries and the
#   libuct_cuda/libucm_cuda modules live under /usr/local, with the modules
#   intentionally installed in /usr/local/lib/ucx.
# - `builder` admits one selected Velox source tree, builds Spark-Gluten's
#   source-owned Arrow 15 C++ recipe as one coherent static closure and builds
#   the patched Arrow Java artifacts into /root/.m2 from that same source tree.
#   It delegates the static AWS SDK installation to the Velox tree's
#   `install_aws_deps` helper, writes the existing
#   /usr/local/share/gluten/cudf-build-info marker, and uses Spark-Gluten's
#   public build entrypoint to compile and install only cuDF, UCXX, and NVTX3.
#   The Velox helper owns the AWS SDK version, selected S3,
#   identity-management and S3-CRT components, and their transitive closure.
# - `carrier` starts from the toolchain and copies the builder's installed
#   /usr/local prefix plus its root-owned Maven repository. Source and build
#   trees under /opt are not copied.
#
# Downstream use:
# A consumer uses the carrier with --cudf_source=SYSTEM, the existing strict
# compatibility check enabled, and mismatch rebuild disabled. An S3-capable
# consumer additionally compiles with --enable_s3=ON; the unconditional AWS
# payload does not itself enable S3. The exact full CUDF_COMMIT marker proves
# source alignment only; it does not prove compiler, CUDA, SM, flags, patches,
# ABI, or binary equivalence.
#
# Inputs and validation:
# BASE_IMAGE selects the compatible CentOS/CUDA toolchain; UCX_VERSION follows
# the selected Velox adapter pin; CUDF_COMMIT, CUDF_VERSION, CUDA_ARCH, and
# NUM_THREADS control the dependency build. The selected Velox source owns the
# AWS SDK pin and installation recipe. Spark-Gluten owns the Arrow 15 C++ and
# Java recipe and patches. GPU-independent image assembly checks the marker and
# packages, the complete static Arrow closure, the five required patched Arrow
# Java artifacts, compiles and links a small S3/S3-CRT consumer against the
# static AWS closure, validates CUDA UCX build configuration and modules, and
# rejects forbidden artifacts. The producer's GPU-enabled post-build validator
# then requires cuda_copy and cuda_ipc from `ucx_info -d` before accepting the
# tag.
#
# Ownership boundaries:
# This recipe does not log in, publish, promote, register, or define catalog,
# request/receipt, portable recipe-identity, or cross-process reuse contracts.
# It also does not copy host CUDA driver libraries or stubs; the NVIDIA
# container runtime owns driver injection. Those concerns remain outside this
# upstream producer. AWS credentials, endpoints, buckets, live S3 requests,
# and downstream S3 enablement and runtime qualification are likewise consumer
# or deployment concerns.

ARG BASE_IMAGE=ghcr.io/facebookincubator/velox-dev:adapters
ARG UCX_VERSION=1.20.1
ARG UCX_SHA256=545c419a7b5e04643cb8bff5a19b3b5071a8f8f0605f1e8efb36f8f3d7bfb9d3

FROM ${BASE_IMAGE} AS toolchain

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Velox's CMake 4 toolchain contract keeps older dependency policies admissible.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH=/usr/lib/jvm/java-17-openjdk/bin:${PATH}

ARG CURL_VERSION=8.12.1
ARG UCX_VERSION
ARG UCX_SHA256
ARG NUM_THREADS

# Match Spark-Gluten's existing cuDF recipe: the inherited CC/CXX variables
# name gcc-toolset-12, so keep that path aligned with the selected GCC 14
# toolchain. UCX is built with CUDA below, and cuDF 26.08 requires CMake 4.
RUN rm -rf /opt/rh/gcc-toolset-12 \
    && ln -s /opt/rh/gcc-toolset-14 /opt/rh/gcc-toolset-12 \
    && dnf install -y java-17-openjdk-devel maven patchelf rdma-core-devel \
    && dnf clean all \
    && UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin \
      uv tool install --force cmake@4.3.2 \
    && chmod -R a+rX /opt/uv-tools \
    && cmake_path=$(readlink -f /usr/local/bin/cmake) \
    && [[ "${cmake_path}" == /opt/uv-tools/* ]] \
    && test -x "${cmake_path}" \
    && cmake --version \
    && mvn --version \
    && patchelf --version \
    && java -version 2>&1 | grep -Eq 'version "17([.]|")' \
    && javac -version 2>&1 | grep -Eq '^javac 17([.]|$)'

# KvikIO requires curl >= 7.80 while CentOS Stream 9 supplies 7.76.
RUN curl --retry 5 --retry-delay 2 --retry-max-time 120 -fsSL \
      "https://curl.se/download/curl-${CURL_VERSION}.tar.gz" \
      | tar xz -C /tmp \
    && cmake -S "/tmp/curl-${CURL_VERSION}" -B /tmp/curl-build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_SHARED_LIBS=ON \
      -DCURL_USE_OPENSSL=ON \
      -DBUILD_TESTING=OFF \
      -DCURL_DISABLE_LDAP=ON \
      -DCURL_USE_LIBPSL=OFF \
    && cmake --build /tmp/curl-build -j"$(nproc)" \
    && cmake --install /tmp/curl-build \
    && ldconfig \
    && rm -rf "/tmp/curl-${CURL_VERSION}" /tmp/curl-build

# Distro UCX packages provide only the CPU transports in this environment.
# Build the selected Velox UCX pin with CUDA and fix libdir so the runtime
# module location is stable and independently checkable by consumers.
RUN source /opt/rh/gcc-toolset-14/enable \
    && [[ "${NUM_THREADS}" =~ ^[1-9][0-9]*$ ]] \
    && curl --retry 5 --retry-delay 2 --retry-max-time 120 -fsSL \
      "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" \
      -o "/tmp/ucx-${UCX_VERSION}.tar.gz" \
    && printf '%s  %s\n' "${UCX_SHA256}" "/tmp/ucx-${UCX_VERSION}.tar.gz" \
      | sha256sum --check --strict \
    && tar xzf "/tmp/ucx-${UCX_VERSION}.tar.gz" -C /tmp \
    && mkdir "/tmp/ucx-${UCX_VERSION}/build-linux" \
    && cd "/tmp/ucx-${UCX_VERSION}/build-linux" \
    && ../contrib/configure-release \
      --prefix=/usr/local \
      --libdir=/usr/local/lib \
      --with-sysroot \
      --enable-cma \
      --enable-mt \
      --with-gnu-ld \
      --with-rdmacm \
      --with-verbs \
      --with-cuda=/usr/local/cuda \
      --without-go \
      --without-java \
    && make -j"${NUM_THREADS}" \
    && make install \
    && printf '/usr/local/lib\n' > /etc/ld.so.conf.d/ucx.conf \
    && ldconfig \
    && test -x /usr/local/bin/ucx_info \
    && find /usr/local/lib/ucx -maxdepth 1 -name 'libuct_cuda.so*' -print -quit | grep -q . \
    && find /usr/local/lib/ucx -maxdepth 1 -name 'libucm_cuda.so*' -print -quit | grep -q . \
    && ucx_build=$(/usr/local/bin/ucx_info -b) \
    && grep -Eq '^#define[[:space:]]+HAVE_CUDA[[:space:]]+1$' <<< "${ucx_build}" \
    && grep -Eq '^#define[[:space:]]+UCX_MODULE_SUBDIR[[:space:]]+"ucx"$' <<< "${ucx_build}" \
    && grep -Fq -- '--with-cuda=/usr/local/cuda' <<< "${ucx_build}" \
    && grep -Eq 'ucm_MODULES[[:space:]]+"[^"]*:cuda([:"]|$)' <<< "${ucx_build}" \
    && grep -Eq 'uct_MODULES[[:space:]]+"[^"]*:cuda([:"]|$)' <<< "${ucx_build}" \
    && ucx_config=$(/usr/local/bin/ucx_info -f) \
    && grep -Fq 'UCX_MODULE_DIR=/usr/local/lib/ucx' <<< "${ucx_config}" \
    && rm -rf "/tmp/ucx-${UCX_VERSION}" "/tmp/ucx-${UCX_VERSION}.tar.gz"

# The inherited image contains an incomplete Arrow installation. Remove it in
# the common stage so neither the builder nor carrier can accidentally combine
# those files with Spark-Gluten's source-owned Arrow 15 static closure.
RUN rm -rf /usr/local/include/arrow /usr/local/include/arrow-glib \
      /usr/local/share/arrow /usr/local/share/doc/arrow \
    && if [[ -d /usr/local/share/gdb ]]; then \
      find /usr/local/share/gdb -type f -name 'libarrow*' -delete; \
    fi \
    && for libdir in /usr/local/lib /usr/local/lib64; do \
      if [[ -d "${libdir}" ]]; then \
        find "${libdir}" \( -type f -o -type l \) \
          -name 'libarrow*' -delete; \
        find "${libdir}" -type f -name 'arrow*.pc' -delete; \
        if [[ -d "${libdir}/cmake" ]]; then \
          find "${libdir}/cmake" -mindepth 1 -maxdepth 1 -type d \
            \( -name 'Arrow*' -o -name 'arrow*' \) -exec rm -rf {} +; \
        fi; \
      fi; \
    done \
    && ! find /usr/local/lib /usr/local/lib64 \
      \( -type f -o -type l \) -name 'libarrow*' -print -quit | grep -q .

FROM toolchain AS builder

ARG CUDF_COMMIT
ARG CUDF_VERSION
ARG CUDA_ARCH
ARG NUM_THREADS

RUN [[ "${CUDF_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] \
    && [[ "${CUDF_VERSION}" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]] \
    && [[ "${CUDA_ARCH}" =~ ^[0-9]+(-real|-virtual)?(,[0-9]+(-real|-virtual)?)*$ ]] \
    && [[ "${NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]

# Copy the existing public producer entrypoint, source-metadata verifier, and
# source-owned Arrow 15 recipe. build/mvn and the root POM are the only Maven
# inputs the selective Arrow Java builder needs. The selected Velox source is a
# separate local build context.
COPY --from=gluten dev/builddeps-veloxbe.sh /opt/gluten/dev/builddeps-veloxbe.sh
COPY --from=gluten dev/build-arrow.sh /opt/gluten/dev/build-arrow.sh
COPY --from=gluten \
  dev/build-helper-functions.sh \
  /opt/gluten/dev/build-helper-functions.sh
COPY --from=gluten dev/verify-system-cudf.py /opt/gluten/dev/verify-system-cudf.py
COPY --from=gluten \
  ep/build-velox/src/build-velox.sh \
  /opt/gluten/ep/build-velox/src/build-velox.sh
COPY --from=gluten \
  ep/build-velox/src/modify_arrow.patch \
  /opt/gluten/ep/build-velox/src/modify_arrow.patch
COPY --from=gluten \
  ep/build-velox/src/modify_arrow_dataset_scan_option.patch \
  /opt/gluten/ep/build-velox/src/modify_arrow_dataset_scan_option.patch
COPY --from=gluten \
  ep/build-velox/src/cmake-compatibility.patch \
  /opt/gluten/ep/build-velox/src/cmake-compatibility.patch
COPY --from=gluten \
  ep/build-velox/src/support_ibm_power.patch \
  /opt/gluten/ep/build-velox/src/support_ibm_power.patch
COPY --from=gluten build/mvn /opt/gluten/build/mvn
COPY --from=gluten pom.xml /opt/gluten/pom.xml
COPY --from=velox . /opt/velox

# Build one coherent Spark-Gluten-owned Arrow C++ static closure and the patched
# Arrow Java artifacts from the same prepared source tree. Clear any inherited
# Arrow Maven state first so missing prepared artifacts cannot be masked by the
# base image. Remove Maven's failed-resolution markers after the successful
# selective build; they are not prepared artifacts and carrier admission rejects
# them. When supplied, the caller's Maven settings file is available only to this
# BuildKit-secret-backed Arrow Java step; it is removed within the same layer and
# is not carrier content. `patch` and the Arrow source/build tree remain
# builder-only.
RUN --mount=type=secret,id=maven_settings \
    set -euo pipefail; \
    remove_maven_settings=OFF; \
    if [[ -f /run/secrets/maven_settings ]]; then \
      mkdir -p /root/.m2; \
      install -m 0600 /run/secrets/maven_settings /root/.m2/settings.xml; \
      remove_maven_settings=ON; \
    fi; \
    trap 'if [[ "${remove_maven_settings}" == ON ]]; then rm -f /root/.m2/settings.xml; fi' EXIT; \
    dnf install -y patch \
    && dnf clean all \
    && source /opt/rh/gcc-toolset-14/enable \
    && cd /opt/gluten \
    && rm -rf /root/.m2/repository/org/apache/arrow \
    && BUILD_ARROW_JAVA=ON CMAKE_BUILD_PARALLEL_LEVEL="${NUM_THREADS}" \
      INSTALL_PREFIX=/usr/local ./dev/build-arrow.sh \
    && find /root/.m2/repository/org/apache/arrow \
      -type f -name '*.lastUpdated' -delete \
    && arrow_archive=$(find /usr/local/lib /usr/local/lib64 -maxdepth 1 \
      -type f -name libarrow.a -print -quit) \
    && bundled_archive=$(find /usr/local/lib /usr/local/lib64 -maxdepth 1 \
      -type f -name libarrow_bundled_dependencies.a -print -quit) \
    && test -n "${arrow_archive}" \
    && test -n "${bundled_archive}" \
    && [[ "$(dirname "${arrow_archive}")" == "$(dirname "${bundled_archive}")" ]] \
    && test -f /usr/local/include/arrow/c/abi.h \
    && test -f /usr/local/include/arrow/c/bridge.h \
    && ! find /usr/local/lib /usr/local/lib64 \
      \( -type f -o -type l \) -name 'libarrow.so*' -print -quit | grep -q . \
    && rm -rf /opt/gluten/ep/_ep/arrow_ep \
    && if [[ "${remove_maven_settings}" == ON ]]; then \
      rm -f /root/.m2/settings.xml; \
      remove_maven_settings=OFF; \
    fi

RUN mkdir -p /usr/local/share/gluten \
    && printf 'CUDF_COMMIT=%s\nCUDF_VERSION=%s\n' \
      "${CUDF_COMMIT}" "${CUDF_VERSION}" \
      > /usr/local/share/gluten/cudf-build-info \
    && python3 /opt/gluten/dev/verify-system-cudf.py \
      --velox-home /opt/velox \
      --version-info /usr/local/share/gluten/cudf-build-info

# This is deliberately a dependency-only selective build through the existing
# Gluten entrypoint. Tests, examples, benchmarks, and consumer libraries remain
# disabled; only the cuDF and UCXX dependency targets are compiled. The selected
# Velox tree's UCX exchange requests UCXX in config mode, so expose the package
# generated earlier in this same configure pass through its normal search path.
RUN source /opt/rh/gcc-toolset-14/enable \
    && export TARGETS="cudf ucxx" \
    && export CMAKE_PREFIX_PATH="/opt/velox/_build/release/_deps/ucxx-build${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
    && cd /opt/gluten \
    && ./dev/builddeps-veloxbe.sh \
      --run_setup_script=OFF \
      --build_arrow=OFF \
      --build_tests=OFF \
      --build_examples=OFF \
      --build_benchmarks=OFF \
      --build_velox_tests=OFF \
      --build_velox_benchmarks=OFF \
      --enable_gpu=ON \
      --cudf_source=BUNDLED \
      --velox_home=/opt/velox \
      --cuda_arch="${CUDA_ARCH}" \
      --num_threads="${NUM_THREADS}" \
      build_velox

# build-velox.sh installs cuDF. Install UCXX and NVTX3 explicitly so the existing
# SYSTEM path can resolve the complete exported package set from /usr/local.
RUN deps=/opt/velox/_build/release/_deps \
    && test -f "${deps}/cudf-build/cmake_install.cmake" \
    && test -f "${deps}/ucxx-build/cmake_install.cmake" \
    && cmake --install "${deps}/cudf-build" --prefix /usr/local \
    && cmake --install "${deps}/ucxx-build" --prefix /usr/local \
    && nvtx_cmake=$(find "${deps}" -path '*/nvtx3-src/c/CMakeLists.txt' -print -quit) \
    && test -n "${nvtx_cmake}" \
    && cmake -S "$(dirname "${nvtx_cmake}")" -B /tmp/nvtx3-install -GNinja \
      -DNVTX3_INSTALL=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --install /tmp/nvtx3-install \
    && find /usr/local -name cudf-config.cmake -print -quit | grep -q . \
    && find /usr/local -name ucxx-config.cmake -print -quit | grep -q . \
    && find /usr/local -name nvtx3-config.cmake -print -quit | grep -q . \
    && find /usr/local -name 'libcudf.so*' -print -quit | grep -q . \
    && find /usr/local -name 'libucxx.so*' -print -quit | grep -q . \
    && rm -rf /tmp/nvtx3-install /root/.cache/ccache

# Install the selected Velox source's complete static AWS S3 dependency
# closure. This deliberately calls Velox's public helper so its source-pinned
# AWS SDK version, component selection (including S3-CRT), and transitive CRT
# recipe cannot drift into a second Spark-Gluten-owned definition. Keeping this
# independent layer after the cuDF/RAPIDS install also lets producer builds
# reuse an unchanged dependency layer; consumer build-tree reuse is not implied.
RUN source /opt/rh/gcc-toolset-14/enable \
    && export BUILD_THREADS="${NUM_THREADS}" \
    && export DEPENDENCY_DIR=/opt/velox/_build/aws-sdk-deps \
    && export INSTALL_PREFIX=/usr/local \
    && cd /opt/velox \
    && source scripts/setup-common.sh \
    && install_aws_deps \
    && awssdk_version_config=$(find /usr/local \
      -path '*/cmake/AWSSDK/AWSSDKConfigVersion.cmake' -print -quit) \
    && test -n "${awssdk_version_config}" \
    && grep -Fq "set(PACKAGE_VERSION \"${AWS_SDK_VERSION}\")" \
      "${awssdk_version_config}"

FROM toolchain AS carrier

ARG CUDF_COMMIT
ARG CUDF_VERSION

RUN [[ "${CUDF_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] \
    && [[ "${CUDF_VERSION}" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]]

# Copy the installed dependency prefix and the root-owned Maven repository.
# Clear only an Arrow cache inherited from a custom toolchain base first;
# Docker directory copies merge with an existing destination. Builder source
# and build trees under /opt are intentionally absent here. The caller's
# optional Maven settings were removed in the builder step and are not copied.
COPY --from=builder /usr/local/ /usr/local/
RUN rm -rf /root/.m2/repository/org/apache/arrow
COPY --from=builder /root/.m2/ /root/.m2/
COPY check-cudf-dependency-image-entrypoint.sh /usr/local/bin/
COPY package-check/ /usr/local/share/gluten/cudf-dependency-check/

RUN printf '/usr/local/lib\n/usr/local/lib64\n' > /etc/ld.so.conf.d/cudf.conf \
    && chmod 0755 /usr/local/bin/check-cudf-dependency-image-entrypoint.sh \
    && chmod 0755 /usr/local/share/gluten/cudf-dependency-check \
    && chmod 0644 \
      /usr/local/share/gluten/cudf-dependency-check/CMakeLists.txt \
      /usr/local/share/gluten/cudf-dependency-check/aws-s3-link-probe.cpp \
    && ldconfig \
    && /usr/local/bin/check-cudf-dependency-image-entrypoint.sh \
      "${CUDF_COMMIT}" --require-cuda-transports=OFF

CMD ["/bin/bash"]
