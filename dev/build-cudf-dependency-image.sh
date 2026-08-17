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
# Build one local, reusable SYSTEM-cuDF dependency carrier from a clean selected
# Velox source tree. The producer checks the caller's full CUDF_COMMIT before
# Docker starts, forwards the selected source as an isolated build context, and
# uses cudf_prebuilt.Dockerfile to install JDK 17, CUDA-enabled UCX,
# Spark-Gluten's source-owned Arrow 15 C++ static closure, its patched Arrow Java
# artifacts under /root/.m2, cuDF, UCXX, NVTX3, and the selected Velox source's
# pinned static AWS SDK C++ S3/S3-CRT closure under /usr/local. The AWS closure
# is unconditional carrier content; consumers still choose independently
# whether to compile Velox with S3 enabled.
#
# After the build, the host-side wrapper runs the image's validation entrypoint
# to require the cuDF source marker, dependency packages, coherent static Arrow
# and AWS closures, the five patched Arrow Java JAR/POM pairs, CUDA UCX modules
# and transports, and the absence of compiled Velox or Gluten artifacts. The
# marker proves cuDF source alignment only; it does not describe Arrow or AWS
# content or prove compiler, CUDA, SM, flags, patches, ABI, or binary
# equivalence. This script creates and validates only the requested local tag:
# registry login, publication, promotion, registration, and downstream consumer
# builds remain outside its ownership.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GLUTEN_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
RECIPE_DIR="${SCRIPT_DIR}/cudf-dependency-image"

VELOX_DIR=""
IMAGE=""
CUDF_COMMIT=""
CUDF_VERSION=""
CUDA_ARCH=""
MAVEN_SETTINGS=""
NUM_THREADS="${NUM_THREADS:-$(nproc)}"
BASE_IMAGE="ghcr.io/facebookincubator/velox-dev:adapters"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build and locally validate a dependency-only SYSTEM-cuDF carrier. The carrier
always includes the static AWS S3/S3-CRT dependency closure pinned by the
selected Velox source; no producer option or consumer S3 setting changes that
content. The command never logs in, pushes, publishes, or registers the image.
Final validation requires a working NVIDIA container runtime and GPU so UCX can
expose its CUDA transports.

Required:
  --velox_dir=PATH          Velox source whose cuDF declaration selects the build
  --image=TAG               Local output image tag
  --cudf_commit=SHA         Full CUDF_COMMIT for installed-cuDF source metadata
  --cudf_version=VERSION    Diagnostic version for installed-cuDF source metadata
  --cuda_arch=ARCH          Explicit target architecture list, e.g. 80-real,90-real

Optional:
  --maven_settings=PATH     Caller-supplied Maven settings.xml for Arrow Java
  --num_threads=N           Positive build parallelism (default: ${NUM_THREADS})
  --base_image=IMAGE        Toolchain base (default: ${BASE_IMAGE})
  -h, --help                Show this help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --velox_dir=*|--velox-dir=*)
      VELOX_DIR="${arg#*=}"
      ;;
    --image=*)
      IMAGE="${arg#*=}"
      ;;
    --cudf_commit=*|--cudf-commit=*)
      CUDF_COMMIT="${arg#*=}"
      ;;
    --cudf_version=*|--cudf-version=*)
      CUDF_VERSION="${arg#*=}"
      ;;
    --cuda_arch=*|--cuda-arch=*)
      CUDA_ARCH="${arg#*=}"
      ;;
    --maven_settings=*)
      MAVEN_SETTINGS="${arg#*=}"
      ;;
    --num_threads=*|--num-threads=*)
      NUM_THREADS="${arg#*=}"
      ;;
    --base_image=*|--base-image=*)
      BASE_IMAGE="${arg#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_value() {
  local name=$1
  local value=$2
  if [ -z "$value" ]; then
    echo "ERROR: ${name} is required" >&2
    usage >&2
    exit 2
  fi
}

require_value --velox_dir "$VELOX_DIR"
require_value --image "$IMAGE"
require_value --cudf_commit "$CUDF_COMMIT"
require_value --cudf_version "$CUDF_VERSION"
require_value --cuda_arch "$CUDA_ARCH"

if [[ ! "$CUDF_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: --cudf_commit must be a full 40-character Git SHA" >&2
  exit 2
fi
if [[ ! "$CUDF_VERSION" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]]; then
  echo "ERROR: --cudf_version must be a nonempty diagnostic version token" >&2
  exit 2
fi
if [[ ! "$CUDA_ARCH" =~ ^[0-9]+(-real|-virtual)?(,[0-9]+(-real|-virtual)?)*$ ]]; then
  echo "ERROR: --cuda_arch must be an explicit comma-separated numeric architecture list" >&2
  exit 2
fi
if [[ ! "$NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --num_threads must be a positive integer" >&2
  exit 2
fi
if [[ "$IMAGE" == -* || "$IMAGE" == *[[:space:]]* ]]; then
  echo "ERROR: --image must be a nonempty local Docker tag without whitespace" >&2
  exit 2
fi
if [[ "$BASE_IMAGE" == -* || "$BASE_IMAGE" == *[[:space:]]* ]]; then
  echo "ERROR: --base_image must be a Docker image reference without whitespace" >&2
  exit 2
fi
if [ -z "$BASE_IMAGE" ]; then
  echo "ERROR: --base_image must not be empty" >&2
  exit 2
fi
if [ ! -d "$VELOX_DIR" ]; then
  echo "ERROR: Velox source directory does not exist: ${VELOX_DIR}" >&2
  exit 2
fi
if [ -n "$MAVEN_SETTINGS" ]; then
  maven_settings_input=$MAVEN_SETTINGS
  if [ ! -f "$MAVEN_SETTINGS" ] || [ ! -s "$MAVEN_SETTINGS" ] \
      || [ ! -r "$MAVEN_SETTINGS" ]; then
    echo "ERROR: --maven_settings must name a readable nonempty regular file: ${MAVEN_SETTINGS}" >&2
    exit 2
  fi
  command -v readlink >/dev/null \
    || { echo "ERROR: readlink is required with --maven_settings" >&2; exit 2; }
  if ! MAVEN_SETTINGS=$(readlink -f -- "$maven_settings_input"); then
    echo "ERROR: could not resolve --maven_settings: ${maven_settings_input}" >&2
    exit 2
  fi
fi

VELOX_DIR=$(cd "$VELOX_DIR" && pwd -P)
if [ ! -f "$VELOX_DIR/CMake/resolve_dependency_modules/cudf.cmake" ]; then
  echo "ERROR: selected Velox source has no cuDF dependency declaration" >&2
  exit 2
fi
if [ -e "$VELOX_DIR/_build" ] || [ -L "$VELOX_DIR/_build" ]; then
  echo "ERROR: selected Velox source must not contain caller-local _build state" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "ERROR: docker is required" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }

marker_dir=$(mktemp -d)
trap 'rm -rf "$marker_dir"' EXIT
marker="$marker_dir/cudf-build-info"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=%s\n' "$CUDF_COMMIT" "$CUDF_VERSION" > "$marker"

# Require the source metadata to match before Docker starts so a malformed or
# mismatched commit cannot trigger an expensive dependency build.
PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/verify-system-cudf.py" \
  --velox-home "$VELOX_DIR" \
  --version-info "$marker"

echo "Building local SYSTEM-cuDF dependency carrier"
echo "  image        : ${IMAGE}"
echo "  base image   : ${BASE_IMAGE}"
echo "  Velox source : ${VELOX_DIR}"
echo "  CUDF_COMMIT  : ${CUDF_COMMIT}"
echo "  CUDF_VERSION : ${CUDF_VERSION}"
echo "  CUDA arch    : ${CUDA_ARCH}"
echo "  threads      : ${NUM_THREADS}"
if [ -n "$MAVEN_SETTINGS" ]; then
  echo "  Maven settings: caller supplied as a BuildKit secret"
fi

docker_build_args=(
  build
  --progress=plain
  --build-context "gluten=${GLUTEN_DIR}"
  --build-context "velox=${VELOX_DIR}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "CUDF_COMMIT=${CUDF_COMMIT}"
  --build-arg "CUDF_VERSION=${CUDF_VERSION}"
  --build-arg "CUDA_ARCH=${CUDA_ARCH}"
  --build-arg "NUM_THREADS=${NUM_THREADS}"
)
if [ -n "$MAVEN_SETTINGS" ]; then
  docker_build_args+=(--secret "id=maven_settings,src=${MAVEN_SETTINGS}")
fi
docker "${docker_build_args[@]}" \
  --tag "$IMAGE" \
  --file "$RECIPE_DIR/cudf_prebuilt.Dockerfile" \
  "$RECIPE_DIR"

"$RECIPE_DIR/check-cudf-dependency-image.sh" \
  --image "$IMAGE" \
  --expected-cudf-commit "$CUDF_COMMIT"

echo "Local SYSTEM-cuDF dependency carrier is ready: ${IMAGE}"
echo "Installed-cuDF source metadata: /usr/local/share/gluten/cudf-build-info"
echo "Prepared patched Arrow Java repository: /root/.m2/repository/org/apache/arrow"
