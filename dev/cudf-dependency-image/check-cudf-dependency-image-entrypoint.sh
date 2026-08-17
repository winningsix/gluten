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

# Purpose:
# Provide the carrier's self-contained content admission check. The producer
# runs this entrypoint while assembling the image, and the host wrapper runs the
# same installed file after the image build. CUDF_DEPENDENCY_ROOT also lets the
# fast test suite exercise the exact checks against local filesystem fixtures.
#
# The base check admits one exact full CUDF_COMMIT source marker, installed
# cuDF/UCXX/NVTX3 development content, the Spark-Gluten-owned static Arrow C++
# closure, the five required patched Arrow Java 15.0.0-gluten artifacts in the
# root Maven repository, the Velox-pinned static AWS S3/S3-CRT dependency
# closure, and CUDA-enabled UCX. A small build-only AWS consumer compiles
# S3Client and S3CrtClient and links the installed component closure; it is
# never run and needs no credentials or network access. The Arrow admission
# requires both static archives and representative C Data Interface headers
# while rejecting shared Arrow libraries, then checks the exact nonempty Java
# JAR/POM pairs, JAR readability, and absence of Maven .lastUpdated files. The
# check also requires CUDA build configuration and both modules at the
# intentional /usr/local/lib/ucx module path. Transport checking defaults to ON:
# the caller must provide GPU access, and the check requires cuda_copy and
# cuda_ipc from /usr/local/bin/ucx_info -d. Image assembly uses OFF because it
# has no NVIDIA driver injection; the producer then repeats the installed check
# in strict ON mode with GPU access. The entrypoint also rejects compiled Velox
# or Gluten artifacts. It does not assess compiler, CUDA, SM, flags, patches,
# ABI, binary equivalence, publication, or downstream consumer runtime.

set -euo pipefail

EXPECTED_COMMIT=${1:-}
TRANSPORT_MODE=${2:---require-cuda-transports=ON}
ROOT=${CUDF_DEPENDENCY_ROOT:-/}

if [ "$#" -gt 2 ] || { [ "$TRANSPORT_MODE" != --require-cuda-transports=ON ] \
    && [ "$TRANSPORT_MODE" != --require-cuda-transports=OFF ]; }; then
  echo "ERROR: usage: $(basename "$0") CUDF_COMMIT [--require-cuda-transports=ON|OFF]" >&2
  exit 2
fi
REQUIRE_CUDA_TRANSPORTS=${TRANSPORT_MODE#*=}
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: expected CUDF_COMMIT must be a full 40-character Git SHA" >&2
  exit 2
fi
if [ ! -d "$ROOT" ]; then
  echo "ERROR: content-check root does not exist: ${ROOT}" >&2
  exit 2
fi
ROOT=$(cd "$ROOT" && pwd -P)

root_path() {
  if [ "$ROOT" = / ]; then
    printf '/%s\n' "$1"
  else
    printf '%s/%s\n' "$ROOT" "$1"
  fi
}

PREFIX=$(root_path usr/local)
MARKER=$(root_path usr/local/share/gluten/cudf-build-info)
CHECK_PROJECT=$(root_path usr/local/share/gluten/cudf-dependency-check)
AWS_LINK_PROBE_SOURCE="$CHECK_PROJECT/aws-s3-link-probe.cpp"
UCX_MODULE_DIR=$(root_path usr/local/lib/ucx)
UCX_INFO=$(root_path usr/local/bin/ucx_info)
ARROW_MAVEN_ROOT=$(root_path root/.m2/repository/org/apache/arrow)
GLUTEN_MAVEN_ROOT=$(root_path root/.m2/repository/org/apache/gluten)
ARROW_JAVA_VERSION=15.0.0-gluten

if [ ! -f "$MARKER" ]; then
  echo "ERROR: installed-cuDF source metadata is missing: ${MARKER}" >&2
  exit 1
fi

mapfile -t commits < <(
  sed -n 's/^[[:space:]]*CUDF_COMMIT[[:space:]]*=[[:space:]]*\([^[:space:]]*\)[[:space:]]*$/\1/p' "$MARKER"
)
mapfile -t versions < <(
  sed -n 's/^[[:space:]]*CUDF_VERSION[[:space:]]*=[[:space:]]*\([^[:space:]]*\)[[:space:]]*$/\1/p' "$MARKER"
)
if [ "${#commits[@]}" -ne 1 ] || [[ ! "${commits[0]}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: installed-cuDF source metadata must contain exactly one full CUDF_COMMIT" >&2
  exit 1
fi
if [ "${commits[0],,}" != "${EXPECTED_COMMIT,,}" ]; then
  echo "ERROR: installed-cuDF source metadata does not match expected CUDF_COMMIT" >&2
  exit 1
fi
if [ "${#versions[@]}" -ne 1 ] || [ -z "${versions[0]}" ]; then
  echo "ERROR: installed-cuDF source metadata must contain one diagnostic CUDF_VERSION" >&2
  exit 1
fi

require_content() {
  local description=$1
  local pattern=$2
  if ! find "$PREFIX" -name "$pattern" -print -quit | grep -q .; then
    echo "ERROR: installed ${description} is missing under ${PREFIX}" >&2
    exit 1
  fi
}

require_content "cuDF CMake package" cudf-config.cmake
require_content "UCXX CMake package" ucxx-config.cmake
require_content "NVTX3 CMake package" nvtx3-config.cmake
require_content "libcudf shared library" 'libcudf.so*'
require_content "libucxx shared library" 'libucxx.so*'
require_content "Arrow static library" libarrow.a
require_content "Arrow bundled-dependencies static library" libarrow_bundled_dependencies.a

if find "$PREFIX" -name 'libarrow.so*' -print -quit | grep -q .; then
  echo "ERROR: shared Arrow library must not be present under ${PREFIX}" >&2
  exit 1
fi

for artifact in \
  arrow-memory-unsafe \
  arrow-memory-core \
  arrow-vector \
  arrow-c-data \
  arrow-dataset; do
  artifact_dir="${ARROW_MAVEN_ROOT}/${artifact}/${ARROW_JAVA_VERSION}"
  jar_file="${artifact_dir}/${artifact}-${ARROW_JAVA_VERSION}.jar"
  pom_file="${artifact_dir}/${artifact}-${ARROW_JAVA_VERSION}.pom"
  if [ ! -s "$jar_file" ] || [ ! -s "$pom_file" ]; then
    echo "ERROR: patched Arrow Java JAR/POM pair is missing: ${artifact}:${ARROW_JAVA_VERSION}" >&2
    exit 1
  fi
  if ! jar tf "$jar_file" >/dev/null; then
    echo "ERROR: patched Arrow Java JAR is unreadable: ${jar_file}" >&2
    exit 1
  fi
done
last_updated=$(find "$ARROW_MAVEN_ROOT" -type f -name '*.lastUpdated' -print -quit)
if [ -n "$last_updated" ]; then
  echo "ERROR: stale Maven resolution marker is present: ${last_updated}" >&2
  exit 1
fi

# These canaries retain the static-payload contract. The compile/link probe
# below owns public-header, CMake-component, and transitive-archive admission.
for aws_archive in \
  libaws-cpp-sdk-s3 \
  libaws-cpp-sdk-identity-management \
  libaws-cpp-sdk-s3-crt \
  libaws-crt-cpp \
  libaws-c-s3; do
  require_content "static AWS dependency archive ${aws_archive}.a" "${aws_archive}.a"
done

require_ucx_module() {
  local module=$1
  if [ ! -d "$UCX_MODULE_DIR" ] \
      || ! find "$UCX_MODULE_DIR" -maxdepth 1 -name "${module}.so*" -print -quit \
        | grep -q .; then
    echo "ERROR: CUDA UCX module ${module}.so is missing under ${UCX_MODULE_DIR}" >&2
    exit 1
  fi
}

require_ucx_module libuct_cuda
require_ucx_module libucm_cuda
if [ ! -x "$UCX_INFO" ]; then
  echo "ERROR: CUDA UCX inspection tool is missing or not executable: ${UCX_INFO}" >&2
  exit 1
fi
if ! ucx_build=$(LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$UCX_INFO" -b); then
  echo "ERROR: CUDA UCX build inspection failed: ${UCX_INFO} -b" >&2
  exit 1
fi
grep -Fq -- '--with-cuda=/usr/local/cuda' <<< "$ucx_build" \
  || { echo "ERROR: UCX was not configured with /usr/local/cuda" >&2; exit 1; }
grep -Eq '^#define[[:space:]]+HAVE_CUDA[[:space:]]+1$' <<< "$ucx_build" \
  || { echo "ERROR: UCX build does not declare CUDA support" >&2; exit 1; }
grep -Eq '^#define[[:space:]]+UCX_MODULE_SUBDIR[[:space:]]+"ucx"$' <<< "$ucx_build" \
  || { echo "ERROR: UCX build does not declare the ucx module subdirectory" >&2; exit 1; }
grep -Eq 'ucm_MODULES[[:space:]]+"[^"]*:cuda([:"]|$)' <<< "$ucx_build" \
  || { echo "ERROR: CUDA UCM module is absent from the UCX build" >&2; exit 1; }
grep -Eq 'uct_MODULES[[:space:]]+"[^"]*:cuda([:"]|$)' <<< "$ucx_build" \
  || { echo "ERROR: CUDA UCT module is absent from the UCX build" >&2; exit 1; }
if ! ucx_config=$(LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$UCX_INFO" -f); then
  echo "ERROR: CUDA UCX configuration inspection failed: ${UCX_INFO} -f" >&2
  exit 1
fi
grep -Fq 'UCX_MODULE_DIR=/usr/local/lib/ucx' <<< "$ucx_config" \
  || { echo "ERROR: UCX module path is not /usr/local/lib/ucx" >&2; exit 1; }
if ! ucx_version=$(LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$UCX_INFO" -v); then
  echo "ERROR: CUDA UCX library inspection failed: ${UCX_INFO} -v" >&2
  exit 1
fi
grep -Fq '# Library path: /usr/local/lib/libucs.so.0' <<< "$ucx_version" \
  || { echo "ERROR: ucx_info does not resolve /usr/local/lib/libucs.so.0" >&2; exit 1; }

if [ "$REQUIRE_CUDA_TRANSPORTS" = ON ]; then
  if ! ucx_devices=$(LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$UCX_INFO" -d); then
    echo "ERROR: CUDA UCX transport inspection failed: ${UCX_INFO} -d" >&2
    exit 1
  fi
  for transport in cuda_copy cuda_ipc; do
    if ! grep -Eq "^[[:space:]#]*Transport:[[:space:]]*${transport}([[:space:]]|$)" \
        <<< "$ucx_devices"; then
      echo "ERROR: required CUDA UCX transport is missing: ${transport}" >&2
      exit 1
    fi
  done
fi

test -f "$PREFIX/include/cudf/types.hpp" \
  || { echo "ERROR: representative cuDF header is missing" >&2; exit 1; }
test -f "$PREFIX/include/ucxx/api.h" \
  || { echo "ERROR: representative UCXX header is missing" >&2; exit 1; }
test -f "$PREFIX/include/arrow/c/abi.h" \
  || { echo "ERROR: representative Arrow C ABI header is missing" >&2; exit 1; }
test -f "$PREFIX/include/arrow/c/bridge.h" \
  || { echo "ERROR: representative Arrow C bridge header is missing" >&2; exit 1; }

for forbidden_dir in opt/gluten opt/velox; do
  candidate=$(root_path "$forbidden_dir")
  if [ -e "$candidate" ]; then
    echo "ERROR: dependency carrier contains forbidden consumer source/build tree: ${candidate}" >&2
    exit 1
  fi
done

gluten_maven_jar=""
if [ -d "$GLUTEN_MAVEN_ROOT" ]; then
  gluten_maven_jar=$(find "$GLUTEN_MAVEN_ROOT" -type f -name '*.jar' \
    -print -quit)
fi
if [ -n "$gluten_maven_jar" ]; then
  echo "ERROR: dependency carrier contains a compiled Gluten Maven artifact: ${gluten_maven_jar}" >&2
  exit 1
fi

artifact_prunes=(
  -path '*/proc' -o -path '*/sys' -o -path '*/dev' -o -path '*/run'
)
if [ "$(id -u)" -ne 0 ]; then
  # Filesystem-fixture checks can run without root. Root deliberately omits this
  # extra prune so producer admission still scans every carrier directory.
  artifact_prunes+=(
    -o \( -type d \( ! -readable -o ! -executable \) \)
  )
fi

artifact=$(find "$ROOT" -xdev \( "${artifact_prunes[@]}" \) -prune -o \
  \( -name 'libgluten*.so*' -o -name 'libgluten*.a' \
     -o -name 'libvelox*.so*' -o -name 'libvelox*.a' \
     -o -name 'gluten-*-bundle-*.jar' \
     -o \( -type f -perm /111 \
       \( -iname '*gluten*' -o -iname '*velox*' \) \) \) -print -quit)
if [ -n "$artifact" ]; then
  echo "ERROR: dependency carrier contains a compiled Velox/Gluten artifact: ${artifact}" >&2
  exit 1
fi

if [ ! -f "$CHECK_PROJECT/CMakeLists.txt" ]; then
  echo "ERROR: installed dependency package check is missing" >&2
  exit 1
fi
if [ ! -f "$AWS_LINK_PROBE_SOURCE" ]; then
  echo "ERROR: installed AWS S3 compile/link probe source is missing" >&2
  exit 1
fi
check_build=$(mktemp -d)
trap 'rm -rf "$check_build"' EXIT
if ! cmake -S "$CHECK_PROJECT" -B "$check_build" -GNinja \
    -DCMAKE_PREFIX_PATH="$PREFIX"; then
  echo "ERROR: installed dependency package configuration failed" >&2
  exit 1
fi
if ! cmake --build "$check_build" --target aws_s3_link_probe --parallel 2; then
  echo "ERROR: installed AWS S3/S3-CRT compile/link probe failed" >&2
  exit 1
fi

echo "Dependency image content verified: CUDF_COMMIT=${commits[0]}; CUDF_VERSION=${versions[0]}"
echo "Installed AWS S3/S3-CRT compile/link probe succeeded."
if [ "$REQUIRE_CUDA_TRANSPORTS" = ON ]; then
  echo "Carrier exposes CUDA UCX modules plus cuda_copy and cuda_ipc transports."
else
  echo "Carrier contains CUDA-enabled UCX modules; live transports were not requested."
fi
echo "Carrier contains cuDF/UCXX/NVTX3, the static Arrow closure, patched Arrow" \
  "Java artifacts, the AWS S3/S3-CRT closure, and no compiled Velox/Gluten artifacts."
