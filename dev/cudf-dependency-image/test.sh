#!/bin/bash

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

# Test intent:
# Guard the dependency-carrier contract with a fast, hermetic premerge suite.
# This script validates producer arguments and Docker forwarding, then drives
# the installed checker against representative marker, package, Arrow C++
# static closure, patched Arrow Java, CUDA UCX, AWS S3/S3-CRT closure, and
# forbidden-artifact fixtures. Missing Arrow/AWS content, an unreadable patched
# JAR, stale Maven resolution state, semantic package/probe failures,
# libuct_cuda/libucm_cuda modules, and cuda_copy/cuda_ipc transports are required
# to fail independently.
#
# How it works:
# - Fake Docker and phase-aware CMake isolate shell-policy testing from image
#   builds while proving both configure and compile/link failures propagate.
# - Fake jar and ucx_info commands keep the fixtures independent of a local JDK
#   and GPU while retaining the exact root/JDK17 carrier contract.
# - Static assertions retain the renamed recipe, dependency-only stage/copy
#   boundaries, source-owned Arrow C++/Java build, frozen strict SYSTEM smoke
#   flags, and no-publication boundary.
# Real AWS compilation/linking, NVIDIA runtime injection, and the full Spark 3.5
# buildbundle remain the producer and protected/nightly responsibilities.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PRODUCER=$(cd "$SCRIPT_DIR/.." && pwd -P)/build-cudf-dependency-image.sh
ARROW_BUILDER=$(cd "$SCRIPT_DIR/.." && pwd -P)/build-arrow.sh
BUILDDEPS=$(cd "$SCRIPT_DIR/.." && pwd -P)/builddeps-veloxbe.sh
CHECK_ENTRYPOINT="$SCRIPT_DIR/check-cudf-dependency-image-entrypoint.sh"
CHECK_HOST_WRAPPER="$SCRIPT_DIR/check-cudf-dependency-image.sh"
SMOKE="$SCRIPT_DIR/smoke-system.sh"
RECIPE="$SCRIPT_DIR/cudf_prebuilt.Dockerfile"
DOCKERIGNORE="$SCRIPT_DIR/.dockerignore"
PACKAGE_CHECK_DIR="$SCRIPT_DIR/package-check"
PACKAGE_CHECK="$PACKAGE_CHECK_DIR/CMakeLists.txt"
PACKAGE_PROBE="$PACKAGE_CHECK_DIR/aws-s3-link-probe.cpp"
GLUTEN_VELOX_CMAKE="$SCRIPT_DIR/../../cpp/velox/CMakeLists.txt"
COMMIT=5beaa5954688fcb12236ffb434e192ea2c77db30
OTHER_COMMIT=33320d64c94a64c94bccc5e2c522721e4d275858
TEST_CUDA_ARCH=80-real,90-real
TEST_GLUTEN_HEAD=1111111111111111111111111111111111111111
TEST_VELOX_HEAD=2222222222222222222222222222222222222222
TEST_IMAGE_ID=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
ARROW_JAVA_VERSION=15.0.0-gluten
ARROW_JAVA_ARTIFACTS=(
  arrow-memory-unsafe
  arrow-memory-core
  arrow-vector
  arrow-c-data
  arrow-dataset
)

tmp=$(mktemp -d)
unreadable_dir=""
cleanup() {
  if [ -n "$unreadable_dir" ]; then
    chmod 0700 "$unreadable_dir" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
fake_bin="$tmp/bin"
velox="$tmp/velox"
mkdir -p "$fake_bin" "$velox/CMake/resolve_dependency_modules"

cat > "$velox/CMake/resolve_dependency_modules/cudf.cmake" <<EOF
set(VELOX_cudf_VERSION 26.08 CACHE STRING "cudf version")
set(VELOX_cudf_COMMIT ${COMMIT})
EOF

cat > "$fake_bin/docker" <<'EOF'
#!/bin/bash
{
  echo CALL
  printf '<%s>\n' "$@"
} >> "${FAKE_DOCKER_LOG:?}"
if [ "${1:-}" = image ] && [ "${2:-}" = inspect ]; then
  printf '%s\n' "${FAKE_DOCKER_IMAGE_ID:?}"
fi
EOF
cat > "$fake_bin/cmake" <<'EOF'
#!/bin/bash
phase=CONFIGURE
if [ "${1:-}" = --build ]; then
  phase=BUILD
fi
{
  echo "$phase"
  printf '<%s>\n' "$@"
} >> "${FAKE_CMAKE_LOG:?}"
if [ "$phase" = CONFIGURE ] && [ "${FAKE_CMAKE_CONFIGURE_FAIL:-OFF}" = ON ]; then
  exit 1
fi
if [ "$phase" = BUILD ] && [ "${FAKE_CMAKE_BUILD_FAIL:-OFF}" = ON ]; then
  exit 1
fi
exit 0
EOF
cat > "$fake_bin/jar" <<'EOF'
#!/bin/bash
test "$#" -eq 2
test "$1" = tf
grep -Fqx 'readable jar fixture' "$2"
EOF
chmod +x "$fake_bin/docker" "$fake_bin/cmake" "$fake_bin/jar"
export PATH="$fake_bin:$PATH"
export FAKE_DOCKER_LOG="$tmp/docker.log"
export FAKE_CMAKE_LOG="$tmp/cmake.log"
export FAKE_DOCKER_IMAGE_ID="$TEST_IMAGE_ID"

expect_failure() {
  local name=$1
  shift
  if "$@" >"$tmp/${name}.out" 2>&1; then
    echo "ERROR: expected failure: ${name}" >&2
    exit 1
  fi
}

common=(
  --velox_dir="$velox"
  --image=local/gluten-cudf-dependencies:test
  --cudf_commit="$COMMIT"
  --cudf_version=26.08
  --cuda_arch="$TEST_CUDA_ARCH"
  --num_threads=4
)

maven_settings="$tmp/maven-settings.xml"
maven_settings_link="$tmp/maven-settings-link.xml"
empty_maven_settings="$tmp/empty-maven-settings.xml"
unreadable_maven_settings="$tmp/unreadable-maven-settings.xml"
nonregular_maven_settings="$tmp/maven-settings-directory"
printf '<settings>caller-secret-fixture</settings>\n' > "$maven_settings"
ln -s "$maven_settings" "$maven_settings_link"
: > "$empty_maven_settings"
printf '<settings>unreadable-fixture</settings>\n' > "$unreadable_maven_settings"
chmod 000 "$unreadable_maven_settings"
mkdir "$nonregular_maven_settings"

: > "$FAKE_DOCKER_LOG"
expect_failure bad_arrow_java env BUILD_ARROW_JAVA=MAYBE "$ARROW_BUILDER"
grep -Fq 'BUILD_ARROW_JAVA must be ON or OFF' "$tmp/bad_arrow_java.out"
expect_failure missing_commit "$PRODUCER" \
  --velox_dir="$velox" --image=local/test:m3 --cudf_version=26.08 \
  --cuda_arch="$TEST_CUDA_ARCH"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure short_commit "$PRODUCER" "${common[@]/--cudf_commit=$COMMIT/--cudf_commit=5beaa595}"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure mismatch "$PRODUCER" "${common[@]/--cudf_commit=$COMMIT/--cudf_commit=$OTHER_COMMIT}"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure bad_arch "$PRODUCER" \
  "${common[@]/--cuda_arch=$TEST_CUDA_ARCH/--cuda_arch=native}"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure bad_threads "$PRODUCER" "${common[@]/--num_threads=4/--num_threads=0}"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure empty_base "$PRODUCER" "${common[@]}" --base_image=
test ! -s "$FAKE_DOCKER_LOG"
expect_failure missing_maven_settings "$PRODUCER" "${common[@]}" \
  --maven_settings="$tmp/does-not-exist.xml"
grep -Fq -- '--maven_settings must name a readable nonempty regular file' \
  "$tmp/missing_maven_settings.out"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure empty_maven_settings "$PRODUCER" "${common[@]}" \
  --maven_settings="$empty_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure nonregular_maven_settings "$PRODUCER" "${common[@]}" \
  --maven_settings="$nonregular_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure unreadable_maven_settings "$PRODUCER" "${common[@]}" \
  --maven_settings="$unreadable_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"
mkdir "$velox/_build"
expect_failure existing_build_state "$PRODUCER" "${common[@]}"
test ! -s "$FAKE_DOCKER_LOG"
rmdir "$velox/_build"

"$PRODUCER" "${common[@]}" > "$tmp/producer.out"
test "$(grep -c '^CALL$' "$FAKE_DOCKER_LOG")" -eq 2
if grep -Fqx '<--secret>' "$FAKE_DOCKER_LOG"; then
  echo "ERROR: default producer unexpectedly forwards Maven settings" >&2
  exit 1
fi
grep -Fqx '<build>' "$FAKE_DOCKER_LOG"
grep -Fqx "<gluten=$(cd "$SCRIPT_DIR/../.." && pwd -P)>" "$FAKE_DOCKER_LOG"
grep -Fqx "<velox=$velox>" "$FAKE_DOCKER_LOG"
grep -Fqx '<BASE_IMAGE=ghcr.io/facebookincubator/velox-dev:adapters>' "$FAKE_DOCKER_LOG"
grep -Fqx "<CUDF_COMMIT=$COMMIT>" "$FAKE_DOCKER_LOG"
grep -Fqx '<CUDF_VERSION=26.08>' "$FAKE_DOCKER_LOG"
grep -Fqx "<CUDA_ARCH=$TEST_CUDA_ARCH>" "$FAKE_DOCKER_LOG"
grep -Fqx '<NUM_THREADS=4>' "$FAKE_DOCKER_LOG"
grep -Fqx '<--file>' "$FAKE_DOCKER_LOG"
grep -Fqx "<$RECIPE>" "$FAKE_DOCKER_LOG"
grep -Fqx '<run>' "$FAKE_DOCKER_LOG"
grep -Fqx '<--user>' "$FAKE_DOCKER_LOG"
grep -Fqx '<0:0>' "$FAKE_DOCKER_LOG"
if grep -Fqx '<65532:65532>' "$FAKE_DOCKER_LOG"; then
  echo "ERROR: carrier validation exposes an unsupported non-root Maven path" >&2
  exit 1
fi
grep -Fqx '<--env>' "$FAKE_DOCKER_LOG"
grep -Fqx '<HOME=/root>' "$FAKE_DOCKER_LOG"
grep -Fqx '<--gpus>' "$FAKE_DOCKER_LOG"
grep -Fqx '<all>' "$FAKE_DOCKER_LOG"
grep -Fqx '</usr/local/bin/check-cudf-dependency-image-entrypoint.sh>' \
  "$FAKE_DOCKER_LOG"
grep -Fqx '<--require-cuda-transports=ON>' "$FAKE_DOCKER_LOG"

: > "$FAKE_DOCKER_LOG"
"$PRODUCER" "${common[@]}" \
  --maven_settings="$maven_settings_link" > "$tmp/producer-maven-settings.out"
test "$(grep -c '^CALL$' "$FAKE_DOCKER_LOG")" -eq 2
grep -Fqx '<--secret>' "$FAKE_DOCKER_LOG"
grep -Fqx \
  "<id=maven_settings,src=$(readlink -f -- "$maven_settings")>" \
  "$FAKE_DOCKER_LOG"
grep -Fqx '  Maven settings: caller supplied as a BuildKit secret' \
  "$tmp/producer-maven-settings.out"
if grep -Fq 'caller-secret-fixture' \
    "$tmp/producer-maven-settings.out" "$FAKE_DOCKER_LOG"; then
  echo "ERROR: producer logged caller Maven settings contents" >&2
  exit 1
fi

root="$tmp/root"
mkdir -p \
  "$root/root/.m2/repository/org/apache/arrow" \
  "$root/usr/local/share/gluten/cudf-dependency-check" \
  "$root/usr/local/lib64/cmake/cudf" \
  "$root/usr/local/lib64/cmake/ucxx" \
  "$root/usr/local/lib64/cmake/nvtx3" \
  "$root/usr/local/include/arrow/c" \
  "$root/usr/local/include/cudf" \
  "$root/usr/local/include/ucxx" \
  "$root/usr/local/bin" \
  "$root/usr/local/lib/ucx" \
  "$root/usr/local/lib64"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$COMMIT" \
  > "$root/usr/local/share/gluten/cudf-build-info"
cp "$PACKAGE_CHECK" "$root/usr/local/share/gluten/cudf-dependency-check/CMakeLists.txt"
cp "$PACKAGE_PROBE" "$root/usr/local/share/gluten/cudf-dependency-check/aws-s3-link-probe.cpp"
touch \
  "$root/usr/local/include/arrow/c/abi.h" \
  "$root/usr/local/include/arrow/c/bridge.h" \
  "$root/usr/local/include/cudf/types.hpp" \
  "$root/usr/local/include/ucxx/api.h" \
  "$root/usr/local/lib64/cmake/cudf/cudf-config.cmake" \
  "$root/usr/local/lib64/cmake/ucxx/ucxx-config.cmake" \
  "$root/usr/local/lib64/cmake/nvtx3/nvtx3-config.cmake" \
  "$root/usr/local/lib64/libcudf.so" \
  "$root/usr/local/lib64/libucxx.so" \
  "$root/usr/local/lib64/libarrow.a" \
  "$root/usr/local/lib64/libarrow_bundled_dependencies.a" \
  "$root/usr/local/lib64/libaws-cpp-sdk-s3.a" \
  "$root/usr/local/lib64/libaws-cpp-sdk-identity-management.a" \
  "$root/usr/local/lib64/libaws-cpp-sdk-s3-crt.a" \
  "$root/usr/local/lib64/libaws-crt-cpp.a" \
  "$root/usr/local/lib64/libaws-c-s3.a" \
  "$root/usr/local/lib/ucx/libuct_cuda.so" \
  "$root/usr/local/lib/ucx/libucm_cuda.so"
for artifact in "${ARROW_JAVA_ARTIFACTS[@]}"; do
  artifact_dir="$root/root/.m2/repository/org/apache/arrow/$artifact/$ARROW_JAVA_VERSION"
  mkdir -p "$artifact_dir"
  printf '<project>fixture</project>\n' \
    > "$artifact_dir/$artifact-$ARROW_JAVA_VERSION.pom"
  printf 'readable jar fixture\n' \
    > "$artifact_dir/$artifact-$ARROW_JAVA_VERSION.jar"
done

export FAKE_UCX_BUILD_OUTPUT="$tmp/ucx-build.txt"
export FAKE_UCX_CONFIG_OUTPUT="$tmp/ucx-config.txt"
export FAKE_UCX_DEVICE_OUTPUT="$tmp/ucx-devices.txt"
export FAKE_UCX_VERSION_OUTPUT="$tmp/ucx-version.txt"
cat > "$root/usr/local/bin/ucx_info" <<'EOF'
#!/bin/bash
case "${1:-}" in
  -b) cat "${FAKE_UCX_BUILD_OUTPUT:?}" ;;
  -f) cat "${FAKE_UCX_CONFIG_OUTPUT:?}" ;;
  -d) cat "${FAKE_UCX_DEVICE_OUTPUT:?}" ;;
  -v) cat "${FAKE_UCX_VERSION_OUTPUT:?}" ;;
  *) exit 2 ;;
esac
EOF
chmod +x "$root/usr/local/bin/ucx_info"
cat > "$FAKE_UCX_BUILD_OUTPUT" <<'EOF'
#define HAVE_CUDA 1
#define UCX_MODULE_SUBDIR "ucx"
#define UCX_CONFIGURE_FLAGS "--enable-mt --with-cuda=/usr/local/cuda"
#define ucm_MODULES ":cuda"
#define uct_MODULES ":cuda:ib:rdmacm:cma"
EOF
printf 'UCX_MODULE_DIR=/usr/local/lib/ucx\n' > "$FAKE_UCX_CONFIG_OUTPUT"
printf '# Library path: /usr/local/lib/libucs.so.0\n' > "$FAKE_UCX_VERSION_OUTPUT"
printf '#      Transport: cuda_copy\n#      Transport: cuda_ipc\n' \
  > "$FAKE_UCX_DEVICE_OUTPUT"

: > "$FAKE_CMAKE_LOG"
CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT" > "$tmp/content.out"
grep -Fqx CONFIGURE "$FAKE_CMAKE_LOG"
grep -Fqx BUILD "$FAKE_CMAKE_LOG"
grep -Fqx '<-GNinja>' "$FAKE_CMAKE_LOG"
grep -Fqx '<--build>' "$FAKE_CMAKE_LOG"
grep -Fqx '<--target>' "$FAKE_CMAKE_LOG"
grep -Fqx '<aws_s3_link_probe>' "$FAKE_CMAKE_LOG"
grep -Fqx '<--parallel>' "$FAKE_CMAKE_LOG"
grep -Fqx '<2>' "$FAKE_CMAKE_LOG"

unreadable_dir="$root/private"
mkdir "$unreadable_dir"
chmod 0000 "$unreadable_dir"
CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT" \
  --require-cuda-transports=OFF > "$tmp/unreadable-directory.out"
chmod 0700 "$unreadable_dir"
rmdir "$unreadable_dir"
unreadable_dir=""

expect_missing_content() {
  local name=$1
  local path=$2
  mv "$path" "$tmp/${name}"
  expect_failure "$name" env CUDF_DEPENDENCY_ROOT="$root" \
    "$CHECK_ENTRYPOINT" "$COMMIT"
  mv "$tmp/${name}" "$path"
}

expect_missing_content missing_s3_static_canary \
  "$root/usr/local/lib64/libaws-cpp-sdk-s3.a"
expect_missing_content missing_crt_static_canary \
  "$root/usr/local/lib64/libaws-c-s3.a"
expect_missing_content missing_arrow_static_archive \
  "$root/usr/local/lib64/libarrow.a"
expect_missing_content missing_arrow_bundled_dependencies \
  "$root/usr/local/lib64/libarrow_bundled_dependencies.a"
expect_missing_content missing_arrow_abi_header \
  "$root/usr/local/include/arrow/c/abi.h"
expect_missing_content missing_arrow_bridge_header \
  "$root/usr/local/include/arrow/c/bridge.h"
touch "$root/usr/local/lib64/libarrow.so"
expect_failure inherited_shared_arrow env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
rm "$root/usr/local/lib64/libarrow.so"

arrow_memory_core_dir="$root/root/.m2/repository/org/apache/arrow/arrow-memory-core/$ARROW_JAVA_VERSION"
arrow_memory_core_pom="$arrow_memory_core_dir/arrow-memory-core-$ARROW_JAVA_VERSION.pom"
cp "$arrow_memory_core_pom" "$tmp/arrow-memory-core.pom"
: > "$arrow_memory_core_pom"
expect_failure empty_arrow_java_pom env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
mv "$tmp/arrow-memory-core.pom" "$arrow_memory_core_pom"

arrow_vector_dir="$root/root/.m2/repository/org/apache/arrow/arrow-vector/$ARROW_JAVA_VERSION"
arrow_vector_jar="$arrow_vector_dir/arrow-vector-$ARROW_JAVA_VERSION.jar"
cp "$arrow_vector_jar" "$tmp/arrow-vector.jar"
printf 'not a readable jar fixture\n' > "$arrow_vector_jar"
expect_failure unreadable_arrow_java_jar env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
mv "$tmp/arrow-vector.jar" "$arrow_vector_jar"

touch "$arrow_vector_dir/arrow-vector-$ARROW_JAVA_VERSION.jar.lastUpdated"
expect_failure stale_arrow_java_resolution env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
rm "$arrow_vector_dir/arrow-vector-$ARROW_JAVA_VERSION.jar.lastUpdated"
expect_missing_content missing_aws_probe_source \
  "$root/usr/local/share/gluten/cudf-dependency-check/aws-s3-link-probe.cpp"
expect_failure aws_package_configure_failure env \
  FAKE_CMAKE_CONFIGURE_FAIL=ON CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
grep -Fq 'installed dependency package configuration failed' \
  "$tmp/aws_package_configure_failure.out"
expect_failure aws_probe_build_failure env \
  FAKE_CMAKE_BUILD_FAIL=ON CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
grep -Fq 'installed AWS S3/S3-CRT compile/link probe failed' \
  "$tmp/aws_probe_build_failure.out"

: > "$FAKE_UCX_DEVICE_OUTPUT"
CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT" \
  --require-cuda-transports=OFF > "$tmp/build-content.out"
printf '#      Transport: cuda_copy\n#      Transport: cuda_ipc\n' \
  > "$FAKE_UCX_DEVICE_OUTPUT"
mv "$root/usr/local/lib/ucx/libuct_cuda.so" "$tmp/libuct_cuda.so"
expect_failure missing_uct_cuda env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
mv "$tmp/libuct_cuda.so" "$root/usr/local/lib/ucx/libuct_cuda.so"
mv "$root/usr/local/lib/ucx/libucm_cuda.so" "$tmp/libucm_cuda.so"
expect_failure missing_ucm_cuda env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT"
mv "$tmp/libucm_cuda.so" "$root/usr/local/lib/ucx/libucm_cuda.so"
printf '#      Transport: cuda_ipc\n' > "$FAKE_UCX_DEVICE_OUTPUT"
expect_failure missing_cuda_copy env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT" --require-cuda-transports=ON
printf '#      Transport: cuda_copy\n' > "$FAKE_UCX_DEVICE_OUTPUT"
expect_failure missing_cuda_ipc env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT" --require-cuda-transports=ON
printf '#      Transport: cuda_copy\n#      Transport: cuda_ipc\n' \
  > "$FAKE_UCX_DEVICE_OUTPUT"
expect_failure bad_transport_mode env CUDF_DEPENDENCY_ROOT="$root" \
  "$CHECK_ENTRYPOINT" "$COMMIT" --require-cuda-transports=MAYBE
mv "$root/usr/local/share/gluten/cudf-build-info" "$tmp/cudf-build-info"
expect_failure missing_marker env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
mv "$tmp/cudf-build-info" "$root/usr/local/share/gluten/cudf-build-info"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$OTHER_COMMIT" \
  > "$root/usr/local/share/gluten/cudf-build-info"
expect_failure mismatched_marker env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$COMMIT" \
  > "$root/usr/local/share/gluten/cudf-build-info"
rm "$root/usr/local/include/cudf/types.hpp"
expect_failure missing_cudf_header env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
touch "$root/usr/local/include/cudf/types.hpp"
rm "$root/usr/local/include/ucxx/api.h"
expect_failure missing_ucxx_header env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
touch "$root/usr/local/include/ucxx/api.h"
printf 'CUDF_COMMIT=%s\n' "$COMMIT" >> "$root/usr/local/share/gluten/cudf-build-info"
expect_failure duplicate_marker env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$COMMIT" \
  > "$root/usr/local/share/gluten/cudf-build-info"
touch "$root/usr/local/lib64/libvelox_exec.a"
expect_failure velox_artifact env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
rm "$root/usr/local/lib64/libvelox_exec.a"
cp "$(type -P true)" "$root/usr/local/bin/velox_cudf_config_test"
expect_failure velox_executable env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
rm "$root/usr/local/bin/velox_cudf_config_test"
cp "$(type -P true)" "$root/usr/local/bin/gluten_backend_test"
expect_failure gluten_executable env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
rm "$root/usr/local/bin/gluten_backend_test"
mkdir -p "$root/root/.m2/repository/org/apache/gluten/gluten-core/fixture"
touch "$root/root/.m2/repository/org/apache/gluten/gluten-core/fixture/gluten-core.jar"
expect_failure gluten_maven_jar env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
rm -rf "$root/root/.m2/repository/org/apache/gluten"
mkdir -p "$root/opt/gluten"
expect_failure gluten_tree env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"
rm -rf "$root/opt/gluten"
rm "$root/usr/local/lib64/cmake/ucxx/ucxx-config.cmake"
expect_failure missing_ucxx env CUDF_DEPENDENCY_ROOT="$root" "$CHECK_ENTRYPOINT" "$COMMIT"

test ! -e "$SCRIPT_DIR/Dockerfile"
grep -Fqx '!cudf_prebuilt.Dockerfile' "$DOCKERIGNORE"
grep -Fqx '!package-check/aws-s3-link-probe.cpp' "$DOCKERIGNORE"
grep -Fq 'export TARGETS="cudf ucxx"' "$RECIPE"
grep -Fq 'ghcr.io/facebookincubator/velox-dev:adapters' "$RECIPE"
grep -Fq 'ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk' "$RECIPE"
grep -Fq 'ENV PATH=/usr/lib/jvm/java-17-openjdk/bin:${PATH}' "$RECIPE"
grep -Fq 'java-17-openjdk-devel' "$RECIPE"
grep -Fq 'maven' "$RECIPE"
grep -Fq 'mvn --version' "$RECIPE"
grep -Fq 'patchelf' "$RECIPE"
grep -Fq 'patchelf --version' "$RECIPE"
grep -Fq "grep -Eq 'version \"17([.]|\")'" "$RECIPE"
grep -Fq "grep -Eq '^javac 17([.]|$)'" "$RECIPE"
grep -Fq 'ln -s /opt/rh/gcc-toolset-14 /opt/rh/gcc-toolset-12' "$RECIPE"
grep -Fq 'UV_TOOL_DIR=/opt/uv-tools' "$RECIPE"
grep -Fq 'UV_TOOL_BIN_DIR=/usr/local/bin' "$RECIPE"
grep -Fq 'chmod -R a+rX /opt/uv-tools' "$RECIPE"
grep -Fq '[[ "${cmake_path}" == /opt/uv-tools/* ]]' "$RECIPE"
if grep -Fq '/root/.local/share/uv/tools/cmake' "$RECIPE"; then
  echo "ERROR: carrier CMake still resolves through root's home" >&2
  exit 1
fi
grep -Fq 'ARG UCX_VERSION=1.20.1' "$RECIPE"
grep -Fq '545c419a7b5e04643cb8bff5a19b3b5071a8f8f0605f1e8efb36f8f3d7bfb9d3' \
  "$RECIPE"
grep -Fq -- '--with-cuda=/usr/local/cuda' "$RECIPE"
grep -Fq '/usr/local/lib/ucx' "$RECIPE"
grep -Fq 'libuct_cuda.so*' "$RECIPE"
grep -Fq 'libucm_cuda.so*' "$RECIPE"
grep -Fq -- '--require-cuda-transports=OFF' "$RECIPE"
grep -Fq '/etc/ld.so.conf.d/cudf.conf' "$RECIPE"
grep -Fq '/opt/velox/_build/release/_deps/ucxx-build' "$RECIPE"
grep -Fq 'rm -rf /usr/local/include/arrow' "$RECIPE"
grep -Fq '/usr/local/share/doc/arrow' "$RECIPE"
grep -Fq '/usr/local/share/gdb' "$RECIPE"
grep -Fq -- "-name 'libarrow*' -delete" "$RECIPE"
grep -Fq -- "-name 'Arrow*' -o -name 'arrow*'" "$RECIPE"
grep -Fq 'COPY --from=gluten dev/build-arrow.sh' "$RECIPE"
grep -Fq 'dev/build-helper-functions.sh' "$RECIPE"
grep -Fq 'COPY --from=gluten build/mvn /opt/gluten/build/mvn' "$RECIPE"
grep -Fq 'COPY --from=gluten pom.xml /opt/gluten/pom.xml' "$RECIPE"
for arrow_patch in \
  modify_arrow.patch \
  modify_arrow_dataset_scan_option.patch \
  cmake-compatibility.patch \
  support_ibm_power.patch; do
  grep -Fq "$arrow_patch" "$RECIPE"
done
test "$(grep -Fc 'rm -rf /root/.m2/repository/org/apache/arrow' "$RECIPE")" -eq 2
grep -Fq 'RUN --mount=type=secret,id=maven_settings' "$RECIPE"
grep -Fq 'install -m 0600 /run/secrets/maven_settings /root/.m2/settings.xml' \
  "$RECIPE"
grep -Fq "rm -f /root/.m2/settings.xml; fi' EXIT" "$RECIPE"
test "$(grep -Fc 'rm -f /root/.m2/settings.xml' "$RECIPE")" -eq 2
if grep -Eq '^COPY .*maven.settings|^COPY .*settings[.]xml' "$RECIPE"; then
  echo "ERROR: carrier recipe copies caller Maven settings into an image layer" >&2
  exit 1
fi
secret_mount_line=$(grep -nF 'RUN --mount=type=secret,id=maven_settings' \
  "$RECIPE" | cut -d: -f1)
arrow_build_line=$(grep -nF \
  'BUILD_ARROW_JAVA=ON CMAKE_BUILD_PARALLEL_LEVEL="${NUM_THREADS}"' \
  "$RECIPE" | cut -d: -f1)
settings_cleanup_line=$(grep -nF 'remove_maven_settings=OFF;' "$RECIPE" \
  | tail -1 | cut -d: -f1)
carrier_maven_copy_line=$(grep -nF \
  'COPY --from=builder /root/.m2/ /root/.m2/' "$RECIPE" | cut -d: -f1)
test "$secret_mount_line" -lt "$arrow_build_line"
test "$settings_cleanup_line" -lt "$carrier_maven_copy_line"
grep -Fq 'BUILD_ARROW_JAVA=ON CMAKE_BUILD_PARALLEL_LEVEL="${NUM_THREADS}"' \
  "$RECIPE"
grep -Fq 'INSTALL_PREFIX=/usr/local ./dev/build-arrow.sh' "$RECIPE"
grep -Fq -- "-type f -name '*.lastUpdated' -delete" "$RECIPE"
grep -Fq 'libarrow_bundled_dependencies.a' "$RECIPE"
grep -Fq '/usr/local/include/arrow/c/abi.h' "$RECIPE"
grep -Fq '/usr/local/include/arrow/c/bridge.h' "$RECIPE"
grep -Fq -- "-name 'libarrow.so*'" "$RECIPE"
grep -Fq 'VELOX_ARROW_BUILD_VERSION=15.0.0' "$ARROW_BUILDER"
grep -Fq 'BUILD_ARROW_JAVA=${BUILD_ARROW_JAVA:-"ON"}' "$ARROW_BUILDER"
grep -Fq 'if [[ "${BUILD_ARROW_JAVA}" == "ON" ]]' "$ARROW_BUILDER"
if grep -Eq -- '--build[_-]arrow[_-]java' \
    "$PRODUCER" "$BUILDDEPS" "$RECIPE"; then
  echo "ERROR: carrier exposes an out-of-scope public Arrow Java option" >&2
  exit 1
fi
grep -Fq -- '--build_tests=OFF' "$RECIPE"
grep -Fq -- '--build_benchmarks=OFF' "$RECIPE"
grep -Fq 'source scripts/setup-common.sh' "$RECIPE"
grep -Fq 'install_aws_deps' "$RECIPE"
grep -Fq '${AWS_SDK_VERSION}' "$RECIPE"
if grep -Eq 'ARG[[:space:]]+AWS_SDK_VERSION|BUILD_ONLY.*s3|git clone.*aws-sdk-cpp' \
    "$RECIPE"; then
  echo "ERROR: recipe duplicates the selected Velox AWS dependency definition" >&2
  exit 1
fi
grep -Fq 'COPY --from=builder /usr/local/ /usr/local/' "$RECIPE"
grep -Fq 'COPY --from=builder /root/.m2/ /root/.m2/' "$RECIPE"
grep -Fq 'COPY check-cudf-dependency-image-entrypoint.sh /usr/local/bin/' \
  "$RECIPE"
grep -Fq 'COPY package-check/ /usr/local/share/gluten/cudf-dependency-check/' \
  "$RECIPE"
grep -Fq 'chmod 0755 /usr/local/share/gluten/cudf-dependency-check' "$RECIPE"
grep -Fq '&& chmod 0644 \' "$RECIPE"
grep -Fq '/usr/local/share/gluten/cudf-dependency-check/CMakeLists.txt \' "$RECIPE"
grep -Fq '/usr/local/share/gluten/cudf-dependency-check/aws-s3-link-probe.cpp \' \
  "$RECIPE"
if grep -Fq 'chmod -R a+rX /usr/local/share/gluten/cudf-dependency-check' "$RECIPE"; then
  echo "ERROR: package-check permissions must use explicit fixed modes" >&2
  exit 1
fi
for aws_canary in \
  libaws-cpp-sdk-s3 \
  libaws-cpp-sdk-identity-management \
  libaws-cpp-sdk-s3-crt \
  libaws-crt-cpp \
  libaws-c-s3; do
  grep -Fq "$aws_canary" "$CHECK_ENTRYPOINT"
done
if grep -Eq 'libaws-c-(auth|cal|common|compression|event-stream|http|io|mqtt|sdkutils)\.a|libaws-checksums\.a|libs2n\.a' \
    "$RECIPE" "$CHECK_ENTRYPOINT"; then
  echo "ERROR: AWS admission duplicates the transitive archive inventory" >&2
  exit 1
fi
grep -Fq 'find_package(ZLIB REQUIRED)' "$PACKAGE_CHECK"
grep -Fq 'find_package(CURL CONFIG REQUIRED)' "$PACKAGE_CHECK"
grep -Fq \
  'find_package(AWSSDK CONFIG REQUIRED COMPONENTS s3 identity-management s3-crt)' \
  "$PACKAGE_CHECK"
curl_package_line=$(grep -nF 'find_package(CURL CONFIG REQUIRED)' \
  "$PACKAGE_CHECK" | cut -d: -f1)
aws_package_line=$(grep -nF \
  'find_package(AWSSDK CONFIG REQUIRED COMPONENTS s3 identity-management s3-crt)' \
  "$PACKAGE_CHECK" | cut -d: -f1)
test "$curl_package_line" -lt "$aws_package_line"
grep -Fq 'find_package(ZLIB REQUIRED)' "$GLUTEN_VELOX_CMAKE"
grep -Fq 'find_package(CURL CONFIG QUIET)' "$GLUTEN_VELOX_CMAKE"
grep -Fq 'if(NOT TARGET CURL::libcurl)' "$GLUTEN_VELOX_CMAKE"
grep -Fq 'find_package(CURL MODULE REQUIRED)' "$GLUTEN_VELOX_CMAKE"
consumer_zlib_line=$(grep -nF 'find_package(ZLIB REQUIRED)' \
  "$GLUTEN_VELOX_CMAKE" | cut -d: -f1)
consumer_curl_line=$(grep -nF 'find_package(CURL CONFIG QUIET)' \
  "$GLUTEN_VELOX_CMAKE" | cut -d: -f1)
consumer_aws_line=$(grep -nF \
  'find_package(AWSSDK REQUIRED COMPONENTS s3;identity-management;s3-crt)' \
  "$GLUTEN_VELOX_CMAKE" | cut -d: -f1)
test "$consumer_zlib_line" -lt "$consumer_curl_line"
test "$consumer_curl_line" -lt "$consumer_aws_line"
grep -Fq 'add_executable(aws_s3_link_probe aws-s3-link-probe.cpp)' "$PACKAGE_CHECK"
grep -Fq '${AWSSDK_LIBRARIES}' "$PACKAGE_CHECK"
for probe_contract in \
  '<aws/core/Aws.h>' \
  '<aws/identity-management/auth/STSAssumeRoleCredentialsProvider.h>' \
  '<aws/s3/S3Client.h>' \
  '<aws/s3-crt/S3CrtClient.h>' \
  'Aws::S3::S3Client s3;' \
  'Aws::S3Crt::S3CrtClient s3Crt;' \
  'Aws::Auth::STSAssumeRoleCredentialsProvider assumeRole('; do
  grep -Fq "$probe_contract" "$PACKAGE_PROBE"
done
grep -Fq 'cmake --build "$check_build" --target aws_s3_link_probe --parallel 2' \
  "$CHECK_ENTRYPOINT"
for arrow_contract in \
  libarrow.a \
  libarrow_bundled_dependencies.a \
  include/arrow/c/abi.h \
  include/arrow/c/bridge.h \
  'libarrow.so*'; do
  grep -Fq "$arrow_contract" "$CHECK_ENTRYPOINT"
done
for artifact in "${ARROW_JAVA_ARTIFACTS[@]}"; do
  grep -Fq "$artifact" "$CHECK_ENTRYPOINT"
done
grep -Fq 'ARROW_JAVA_VERSION=15.0.0-gluten' "$CHECK_ENTRYPOINT"
grep -Fq 'if [ ! -s "$jar_file" ] || [ ! -s "$pom_file" ]' "$CHECK_ENTRYPOINT"
grep -Fq 'jar tf "$jar_file"' "$CHECK_ENTRYPOINT"
grep -Fq -- "-name '*.lastUpdated'" "$CHECK_ENTRYPOINT"
grep -Fq '! -readable -o ! -executable' "$CHECK_ENTRYPOINT"
if grep -Eq -- '65532:65532|HOME=/tmp' "$CHECK_HOST_WRAPPER"; then
  echo "ERROR: host validation exposes an unsupported non-root Maven path" >&2
  exit 1
fi
grep -Fq -- '--user 0:0' "$CHECK_HOST_WRAPPER"
grep -Fq -- '--env HOME=/root' "$CHECK_HOST_WRAPPER"
grep -Fq -- '--gpus all' "$CHECK_HOST_WRAPPER"
grep -Fq -- '--require-cuda-transports=ON' "$CHECK_HOST_WRAPPER"
if grep -ERn 'docker (login|push)' \
  "$PRODUCER" \
  "$RECIPE" \
  "$CHECK_ENTRYPOINT" \
  "$CHECK_HOST_WRAPPER" \
  "$SCRIPT_DIR/smoke-system.sh"; then
  echo "ERROR: producer includes an out-of-scope publication or identity contract" >&2
  exit 1
fi
if grep -Eq 'MVN_SET|MAVEN_SETTINGS_FILE' "$PRODUCER" "$SMOKE"; then
  echo "ERROR: Maven settings interface must not define an environment alias" >&2
  exit 1
fi

smoke_bin="$tmp/smoke-bin"
mkdir -p "$smoke_bin"
cat > "$smoke_bin/git" <<'EOF'
#!/bin/bash
set -euo pipefail

test "${1:-}" = -C
source_dir=$2
shift 2
case "$source_dir" in
  "${FAKE_GLUTEN_DIR:?}")
    source_kind=gluten
    source_head=${FAKE_GLUTEN_HEAD:?}
    ;;
  "${FAKE_VELOX_DIR:?}")
    source_kind=velox
    source_head=${FAKE_VELOX_HEAD:?}
    ;;
  *) exit 1 ;;
esac
case "${1:-}:${2:-}" in
  rev-parse:--is-inside-work-tree)
    test "${FAKE_NOT_GIT_SOURCE:-}" != "$source_kind"
    echo true
    ;;
  rev-parse:--verify)
    test "${3:-}" = 'HEAD^{commit}'
    printf '%s\n' "$source_head"
    ;;
  status:--porcelain=v1)
    if [ "${FAKE_DIRTY_SOURCE:-}" = "$source_kind" ]; then
      echo ' M dirty-source-fixture'
    fi
    ;;
  archive:--format=tar)
    test "${3:-}" = "$source_head"
    printf 'tracked archive fixture for %s\n' "$source_kind"
    ;;
  *) exit 1 ;;
esac
EOF
cat > "$smoke_bin/nproc" <<'EOF'
#!/bin/bash
printf '37\n'
EOF
chmod +x "$smoke_bin/git" "$smoke_bin/nproc"

smoke_env=(
  env
  "PATH=$smoke_bin:$PATH"
  "FAKE_GLUTEN_DIR=$(cd "$SCRIPT_DIR/../.." && pwd -P)"
  "FAKE_VELOX_DIR=$velox"
  "FAKE_GLUTEN_HEAD=$TEST_GLUTEN_HEAD"
  "FAKE_VELOX_HEAD=$TEST_VELOX_HEAD"
)
smoke_command=(
  "$SMOKE"
  --image=local/gluten-cudf-dependencies:test
  --velox_dir="$velox"
  --cuda_arch="$TEST_CUDA_ARCH"
)

: > "$FAKE_DOCKER_LOG"
expect_failure smoke_invalid_hdfs "${smoke_env[@]}" \
  FAKE_NOT_GIT_SOURCE=gluten "${smoke_command[@]}" --enable_hdfs=MAYBE
grep -Fq -- '--enable_hdfs must be ON or OFF' "$tmp/smoke_invalid_hdfs.out"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure smoke_missing_maven_settings "${smoke_env[@]}" \
  "${smoke_command[@]}" --maven_settings="$tmp/does-not-exist.xml"
grep -Fq -- '--maven_settings must name a readable nonempty regular file' \
  "$tmp/smoke_missing_maven_settings.out"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure smoke_empty_maven_settings "${smoke_env[@]}" \
  "${smoke_command[@]}" --maven_settings="$empty_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure smoke_nonregular_maven_settings "${smoke_env[@]}" \
  "${smoke_command[@]}" --maven_settings="$nonregular_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"
expect_failure smoke_unreadable_maven_settings "${smoke_env[@]}" \
  "${smoke_command[@]}" --maven_settings="$unreadable_maven_settings"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure non_git_gluten_source "${smoke_env[@]}" \
  FAKE_NOT_GIT_SOURCE=gluten "${smoke_command[@]}"
grep -Fq 'ERROR: Gluten source is not a Git worktree' \
  "$tmp/non_git_gluten_source.out"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure dirty_gluten_source "${smoke_env[@]}" \
  FAKE_DIRTY_SOURCE=gluten "${smoke_command[@]}"
grep -Fq 'ERROR: Gluten source must be clean' "$tmp/dirty_gluten_source.out"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure dirty_velox_source "${smoke_env[@]}" \
  FAKE_DIRTY_SOURCE=velox "${smoke_command[@]}"
grep -Fq 'ERROR: Velox source must be clean' "$tmp/dirty_velox_source.out"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure short_gluten_head "${smoke_env[@]}" \
  FAKE_GLUTEN_HEAD=11111111 "${smoke_command[@]}"
grep -Fq 'ERROR: Gluten source HEAD is not a full SHA' \
  "$tmp/short_gluten_head.out"
test ! -s "$FAKE_DOCKER_LOG"

expect_failure invalid_image_id "${smoke_env[@]}" \
  FAKE_DOCKER_IMAGE_ID=local-image-id "${smoke_command[@]}"
grep -Fq 'ERROR: dependency carrier did not resolve to one full image ID' \
  "$tmp/invalid_image_id.out"
test "$(grep -c '^CALL$' "$FAKE_DOCKER_LOG")" -eq 1
if grep -Fqx '<run>' "$FAKE_DOCKER_LOG"; then
  echo "ERROR: protected smoke ran an unresolved carrier image" >&2
  exit 1
fi

: > "$FAKE_DOCKER_LOG"
"${smoke_env[@]}" "${smoke_command[@]}" > "$tmp/smoke.out"
if printf '%s\n' "${smoke_command[@]}" \
    | grep -Eq '^--(enable_hdfs|num_threads)='; then
  echo "ERROR: canonical smoke command no longer relies on HDFS OFF and all-core defaults" >&2
  exit 1
fi
grep -Fqx "Smoke Gluten source HEAD: $TEST_GLUTEN_HEAD" "$tmp/smoke.out"
grep -Fqx "Smoke Velox source HEAD: $TEST_VELOX_HEAD" "$tmp/smoke.out"
grep -Fqx \
  "Smoke dependency carrier image ID: local/gluten-cudf-dependencies:test -> $TEST_IMAGE_ID" \
  "$tmp/smoke.out"
grep -Fq 'carrier-base glibc/loader and JDK 17; host-injected NVIDIA driver' \
  "$tmp/smoke.out"
grep -Fqx 'Smoke HDFS build mode: OFF' "$tmp/smoke.out"
grep -Fqx '<SMOKE_ENABLE_HDFS=OFF>' "$FAKE_DOCKER_LOG"
grep -Fqx '<SMOKE_NUM_THREADS=37>' "$FAKE_DOCKER_LOG"
if grep -Fq 'dst=/root/.m2/settings.xml' "$FAKE_DOCKER_LOG"; then
  echo "ERROR: canonical smoke unexpectedly mounts Maven settings" >&2
  exit 1
fi

: > "$FAKE_DOCKER_LOG"
"${smoke_env[@]}" "${smoke_command[@]}" --enable_hdfs=OFF --num_threads=4 \
  > "$tmp/smoke-hdfs-off.out"
grep -Fqx 'Smoke HDFS build mode: OFF' "$tmp/smoke-hdfs-off.out"
grep -Fqx '<SMOKE_ENABLE_HDFS=OFF>' "$FAKE_DOCKER_LOG"
grep -Fqx '<SMOKE_NUM_THREADS=4>' "$FAKE_DOCKER_LOG"

: > "$FAKE_DOCKER_LOG"
"${smoke_env[@]}" "${smoke_command[@]}" --enable_hdfs=ON \
  > "$tmp/smoke-hdfs-on.out"
grep -Fqx 'Smoke HDFS build mode: ON' "$tmp/smoke-hdfs-on.out"
grep -Fqx '<SMOKE_ENABLE_HDFS=ON>' "$FAKE_DOCKER_LOG"
grep -Fqx '<SMOKE_NUM_THREADS=37>' "$FAKE_DOCKER_LOG"
grep -Fq -- '--enable_hdfs="${SMOKE_ENABLE_HDFS}"' "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$velox_cache" VELOX_ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$gluten_cache" ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"' \
  "$FAKE_DOCKER_LOG"

: > "$FAKE_DOCKER_LOG"
"${smoke_env[@]}" "${smoke_command[@]}" \
  --maven_settings="$maven_settings_link" > "$tmp/smoke-maven-settings.out"
grep -Fqx 'Smoke Maven settings: caller-supplied file mounted read-only' \
  "$tmp/smoke-maven-settings.out"
grep -Fq \
  "src=$(readlink -f -- "$maven_settings"),dst=/root/.m2/settings.xml,readonly" \
  "$FAKE_DOCKER_LOG"
if grep -Fq 'caller-secret-fixture' \
    "$tmp/smoke-maven-settings.out" "$FAKE_DOCKER_LOG"; then
  echo "ERROR: smoke logged caller Maven settings contents" >&2
  exit 1
fi
grep -Fqx '<image>' "$FAKE_DOCKER_LOG"
grep -Fqx '<inspect>' "$FAKE_DOCKER_LOG"
grep -Fqx '<{{.Id}}>' "$FAKE_DOCKER_LOG"
test "$(grep -Fxc "<$TEST_IMAGE_ID>" "$FAKE_DOCKER_LOG")" -eq 1
test "$(grep -Fxc '<local/gluten-cudf-dependencies:test>' \
  "$FAKE_DOCKER_LOG")" -eq 1
grep -Fq -- '--run_setup_script=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--enable_s3=ON' "$FAKE_DOCKER_LOG"
grep -Fq -- '--enable_hdfs="${SMOKE_ENABLE_HDFS}"' "$FAKE_DOCKER_LOG"
grep -Fqx '<SMOKE_ENABLE_HDFS=OFF>' "$FAKE_DOCKER_LOG"
grep -Fqx '<SMOKE_NUM_THREADS=37>' "$FAKE_DOCKER_LOG"
grep -Fq -- '--cudf_source=SYSTEM' "$FAKE_DOCKER_LOG"
grep -Fq -- '--cudf_version_info=/usr/local/share/gluten/cudf-build-info' "$FAKE_DOCKER_LOG"
grep -Fq -- '--cudf_compatibility_check=ON' "$FAKE_DOCKER_LOG"
grep -Fq -- '--rebuild_if_mismatch=OFF' "$FAKE_DOCKER_LOG"
grep -Fq 'dst=/source/gluten.tar,readonly' "$FAKE_DOCKER_LOG"
grep -Fq 'dst=/source/velox.tar,readonly' "$FAKE_DOCKER_LOG"
grep -Fq 'tar -C /smoke/gluten -xf /source/gluten.tar' "$FAKE_DOCKER_LOG"
grep -Fq 'tar -C /smoke/velox -xf /source/velox.tar' "$FAKE_DOCKER_LOG"
if grep -Eq 'src=.*/,dst=/source/(gluten|velox),readonly|tar -C /source/' \
    "$FAKE_DOCKER_LOG"; then
  echo "ERROR: protected smoke copies a worktree instead of exact HEAD content" >&2
  exit 1
fi
grep -Fq '<--user>' "$FAKE_DOCKER_LOG"
grep -Fq '<0:0>' "$FAKE_DOCKER_LOG"
grep -Fq '<HOME=/root>' "$FAKE_DOCKER_LOG"
grep -Fq 'test "$HOME" = /root' "$FAKE_DOCKER_LOG"
grep -Fq 'test -x "$JAVA_HOME/bin/java"' "$FAKE_DOCKER_LOG"
grep -Fq 'test -x "$JAVA_HOME/bin/javac"' "$FAKE_DOCKER_LOG"
grep -Fq 'Smoke Java runtime: ${java_version}; ${javac_version}' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'export TARGETS="velox velox_cudf_exec"' "$FAKE_DOCKER_LOG"
grep -Fq 'export GLUTEN_BUNDLE_MAVEN_PROFILES=backends-velox,spark-3.5,java-17' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$spark_version" = 3.5.5' "$FAKE_DOCKER_LOG"
grep -Fq 'test "$scala_binary_version" = 2.12' "$FAKE_DOCKER_LOG"
grep -Fq 'Maven consumer versions: Spark ${spark_version}; Scala ${scala_binary_version}' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib' "$FAKE_DOCKER_LOG"
maven_bootstrap_line=$(grep -nF './build/mvn --version' "$FAKE_DOCKER_LOG" \
  | cut -d: -f1)
buildbundle_line=$(grep -nF './dev/buildbundle-veloxbe.sh' "$FAKE_DOCKER_LOG" \
  | cut -d: -f1)
test "$maven_bootstrap_line" -lt "$buildbundle_line"
grep -Fq './dev/buildbundle-veloxbe.sh' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_arrow=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_tests=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_examples=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_benchmarks=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_velox_tests=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--build_velox_benchmarks=OFF' "$FAKE_DOCKER_LOG"
grep -Fq -- '--enable_gpu=ON' "$FAKE_DOCKER_LOG"
grep -Fq -- '--spark_version=3.5' "$FAKE_DOCKER_LOG"
if grep -Eq 'velox_cudf_null_mask_test|velox_s3config_test|--build_velox_tests=ON' \
    "$FAKE_DOCKER_LOG"; then
  echo "ERROR: protected smoke still enables deferred Velox test targets" >&2
  exit 1
fi
grep -Fq 'test "$(cache_value "$velox_cache" cudf_SOURCE)" = SYSTEM' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$velox_cache" VELOX_ENABLE_S3)" = ON' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$velox_cache" VELOX_ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$gluten_cache" cudf_SOURCE)" = SYSTEM' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$gluten_cache" ENABLE_S3)" = ON' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'test "$(cache_value "$gluten_cache" ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'Native feature boundary: S3 compile/link=ON; HDFS build=${SMOKE_ENABLE_HDFS}; Hadoop/libhdfs runtime supplied downstream and not exercised' \
  "$FAKE_DOCKER_LOG"
grep -Fq 'CUDF_COMPATIBILITY_RECONCILED' "$FAKE_DOCKER_LOG"
grep -Fq 'ARROW_LIB_arrow_bundled_dependencies' "$FAKE_DOCKER_LOG"
grep -Fq '/smoke/gluten/cpp/build/releases/libgluten.so' "$FAKE_DOCKER_LOG"
grep -Fq 'ldd "$gluten_library"' "$FAKE_DOCKER_LOG"
grep -Fq 'libgluten.so has unresolved shared dependencies' "$FAKE_DOCKER_LOG"
grep -Fq 'AWSSDK_DIR' "$FAKE_DOCKER_LOG"
grep -Fq '/smoke/gluten/gluten-arrow/target' "$FAKE_DOCKER_LOG"
grep -Fq 'gluten-arrow-*-3.5.jar' "$FAKE_DOCKER_LOG"
grep -Fq 'expected one Spark 3.5 Gluten Arrow JAR' "$FAKE_DOCKER_LOG"
grep -Fq 'jar tf "${gluten_arrow_jars[0]}"' "$FAKE_DOCKER_LOG"
grep -Fq 'check_awssdk_resolution "$velox_cache" Velox' "$FAKE_DOCKER_LOG"
grep -Fq 'check_awssdk_resolution "$gluten_cache" Gluten' "$FAKE_DOCKER_LOG"
grep -Fq '/smoke/gluten/package/target' "$FAKE_DOCKER_LOG"
grep -Fq 'gluten-velox-bundle-spark3.5_2.12-*.jar' "$FAKE_DOCKER_LOG"
grep -Fq 'expected exactly one Spark 3.5 Gluten bundle' "$FAKE_DOCKER_LOG"
grep -Fq 'jar tf "${bundles[0]}"' "$FAKE_DOCKER_LOG"
grep -Fq 'Readable Spark 3.5.5/Scala 2.12 Gluten bundle' "$FAKE_DOCKER_LOG"

echo "SYSTEM-cuDF dependency carrier focused tests passed"
