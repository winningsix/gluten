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

# Build Gluten with local Velox repo inside a cuDF-enabled Docker image.
# Supports centos9 (default) and ubuntu2204.
# Usage: ./dev/docker-build-cudf.sh [options]

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GLUTEN_DIR="${SCRIPT_DIR}/.."
VELOX_DIR="${SCRIPT_DIR}/../../velox"

CONTAINER_NAME="gluten_cudf_build"
SYSTEM="centos9"          # target OS: centos9 (default) or ubuntu2204
DOCKER_IMAGE=""           # derived from SYSTEM unless --image is given explicitly
IMAGE_EXPLICIT=false
SPARK_VERSION="3.5"
BUILD_ARROW="ON"
ENABLE_HDFS="OFF"
ENABLE_S3="OFF"
REBUILD=false   # if true: skip Arrow, clear cmake cache, re-run velox+cpp+mvn only

# Velox source — only used if VELOX_DIR does not exist (fallback clone).
VELOX_REPO="https://gitlab-master.nvidia.com/alfxu/velox.git"
VELOX_BRANCH="alfxu_dev"

# CUDA architecture — resolved interactively if not passed via --cuda_arch
CUDA_ARCH=""
CUDA_ARCH_EXPLICIT=false

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --system=SYSTEM           Target OS for the build container (default: centos9)
                            Supported: centos9, ubuntu2204
  --spark_version=VERSION   Spark version to build for (default: 3.5)
                            Supported: 3.3, 3.4, 3.5, 4.0, 4.1, ALL
  --rebuild                 Incremental rebuild: skip Arrow, clear velox cmake cache,
                            re-run only velox + gluten C++ + Maven
  --container=NAME          Docker container name (default: gluten_cudf_build)
  --image=IMAGE             Docker image to use (overrides --system image selection)
                            centos9 default:    apache/gluten:centos-9-jdk8-cudf
                            ubuntu2204 default: apache/gluten:ubuntu-22.04-jdk8-cudf
  --velox_repo=URL          Velox git repo to clone if velox dir is missing
                            (default: https://gitlab-master.nvidia.com/alfxu/velox.git)
  --velox_branch=BRANCH     Velox branch to clone (default: alfxu_dev)
  --enable_hdfs=ON|OFF      Enable HDFS support (default: OFF)
  --enable_s3=ON|OFF        Enable S3 support (default: OFF)
  --cuda_arch=ARCH          CUDA compute architectures (default: native)
                            native     — auto-detect local GPU
                            all-major  — 70,75,80,86,89,90 (portable)
                            <num>      — specific arch, e.g. 86 for A6000/RTX3090
  -h, --help                Show this help message
EOF
}

for arg in "$@"; do
  case $arg in
    --system=*)        SYSTEM="${arg#*=}" ;;
    --spark_version=*) SPARK_VERSION="${arg#*=}" ;;
    --rebuild)         REBUILD=true ;;
    --container=*)     CONTAINER_NAME="${arg#*=}" ;;
    --image=*)         DOCKER_IMAGE="${arg#*=}"; IMAGE_EXPLICIT=true ;;
    --velox_repo=*)    VELOX_REPO="${arg#*=}" ;;
    --velox_branch=*)  VELOX_BRANCH="${arg#*=}" ;;
    --enable_hdfs=*)   ENABLE_HDFS="${arg#*=}" ;;
    --enable_s3=*)     ENABLE_S3="${arg#*=}" ;;
    --cuda_arch=*)     CUDA_ARCH="${arg#*=}"; CUDA_ARCH_EXPLICIT=true ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

# ── Resolve Docker image from system if not explicitly overridden ─────────────
if [ "$IMAGE_EXPLICIT" = false ]; then
  case "$SYSTEM" in
    centos9)    DOCKER_IMAGE="apache/gluten:centos-9-jdk8-cudf" ;;
    ubuntu2204) DOCKER_IMAGE="apache/gluten:ubuntu-22.04-jdk8-cudf" ;;
    *)
      echo "ERROR: Unknown --system value: '$SYSTEM'. Supported: centos9, ubuntu2204"
      usage; exit 1
      ;;
  esac
fi

if [ "$REBUILD" = true ]; then
  BUILD_ARROW="OFF"
fi

# ── CUDA arch prompt (skipped if --cuda_arch was passed) ─────────────────────
if [ "$CUDA_ARCH_EXPLICIT" = false ]; then
  # Detect local GPU compute capability for a helpful hint
  DETECTED_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d '.' || true)
  if [ -n "$DETECTED_CAP" ]; then
    DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    HINT=" (detected: ${DETECTED_GPU}, sm_${DETECTED_CAP})"
  else
    HINT=""
  fi

  echo ""
  echo "Select CUDA target architecture${HINT}:"
  echo "  1) native     — build only for the local GPU [default]"
  echo "  2) all-major  — build for 70,75,80,86,89,90 (portable)"
  if [ -n "$DETECTED_CAP" ]; then
  echo "  3) ${DETECTED_CAP}        — pin to detected GPU (sm_${DETECTED_CAP})"
  fi
  echo "  *) custom     — enter a value or comma-separated list (e.g. 80,86)"
  echo ""
  read -r -p "Choice [1]: " ARCH_CHOICE
  ARCH_CHOICE="${ARCH_CHOICE:-1}"

  case "$ARCH_CHOICE" in
    1|"")      CUDA_ARCH="native" ;;
    2)         CUDA_ARCH="all-major" ;;
    3)         CUDA_ARCH="${DETECTED_CAP:-native}" ;;
    *)
      # Accept: single arch (86), comma-separated arches (80,86), or named values (native, all-major)
      if [[ "$ARCH_CHOICE" =~ ^[0-9]+(,[0-9]+)*$ ]] || [[ "$ARCH_CHOICE" =~ ^[a-z-]+$ ]]; then
        CUDA_ARCH="$ARCH_CHOICE"
      else
        read -r -p "Enter CUDA arch value (e.g. 86, 80,86, all-major, native): " CUDA_ARCH
      fi
      ;;
  esac
  echo "→ Using CUDA arch: ${CUDA_ARCH}"
  echo ""
fi

# ── Validation ───────────────────────────────────────────────────────────────
if [ ! -d "$GLUTEN_DIR" ]; then
  echo "ERROR: gluten directory not found at $GLUTEN_DIR"
  exit 1
fi
if [ ! -d "$VELOX_DIR" ]; then
  echo "ERROR: velox directory not found at $VELOX_DIR"
  exit 1
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo "=============================================="
echo " Gluten cuDF Build"
echo "=============================================="
echo " System        : $SYSTEM"
echo " Gluten dir    : $GLUTEN_DIR"
echo " Velox dir     : $VELOX_DIR"
echo " Velox repo    : $VELOX_REPO  (branch: $VELOX_BRANCH)"
echo " CUDA arch     : $CUDA_ARCH"
echo " Docker image  : $DOCKER_IMAGE"
echo " Container     : $CONTAINER_NAME"
echo " Spark version : $SPARK_VERSION"
echo " Build Arrow   : $BUILD_ARROW"
echo " Enable HDFS   : $ENABLE_HDFS"
echo " Rebuild mode  : $REBUILD"
echo "=============================================="
echo ""

# ── Step 1: Start container ───────────────────────────────────────────────────
if docker inspect "$CONTAINER_NAME" &>/dev/null; then
  STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME")
  if [ "$STATUS" = "running" ]; then
    echo "[1/5] Container '$CONTAINER_NAME' is already running — reusing it."
  else
    echo "[1/5] Container '$CONTAINER_NAME' exists but is stopped — restarting."
    docker start "$CONTAINER_NAME"
  fi
else
  echo "[1/5] Starting container '$CONTAINER_NAME'..."
  docker run --name "$CONTAINER_NAME" \
    --gpus all \
    -v "${GLUTEN_DIR}:/opt/gluten" \
    -v "${VELOX_DIR}:/opt/velox" \
    -itd \
    "$DOCKER_IMAGE"
  echo "      Container started."
fi

# ── Step 2: Verify GPU ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Verifying GPU access..."
if ! docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>/dev/null; then
  echo "ERROR: nvidia-smi failed. Check that --gpus all is working and nvidia-container-toolkit is configured."
  exit 1
fi

# ── Step 3: Clear cmake cache on rebuild ──────────────────────────────────────
if [ "$REBUILD" = true ]; then
  echo ""
  echo "[3/5] Rebuild mode: clearing velox cmake cache..."
  docker exec "$CONTAINER_NAME" bash -c \
    "rm -f /opt/velox/_build/release/CMakeCache.txt && echo '      CMakeCache.txt removed.'"
else
  echo ""
  echo "[3/5] Full build — skipping cmake cache clear."
fi

# ── Step 4: Run bundle build ───────────────────────────────────────────────────
echo ""
echo "[4/5] Running bundle build (output -> gluten/build.log)..."
echo ""

TEE_FLAG=$([ "$REBUILD" = true ] && echo "-a" || echo "")

docker exec "$CONTAINER_NAME" bash -c "
  cd /opt/gluten && \
  bash ./dev/buildbundle-veloxbe.sh \
    --run_setup_script=OFF \
    --build_arrow=${BUILD_ARROW} \
    --spark_version=${SPARK_VERSION} \
    --enable_gpu=ON \
    --enable_hdfs=${ENABLE_HDFS} \
    --enable_s3=${ENABLE_S3} \
    --velox_home=/opt/velox \
    --velox_repo=${VELOX_REPO} \
    --velox_branch=${VELOX_BRANCH} \
    --cuda_arch=${CUDA_ARCH} \
    2>&1 | tee ${TEE_FLAG} /opt/gluten/build.log
"

# ── Step 5: Build 3rd-party jars ──────────────────────────────────────────────
echo ""
echo "[5/5] Building 3rd-party jars (output -> gluten/thirdparty.log)..."
echo ""

docker exec "$CONTAINER_NAME" bash -c "
  cd /opt/gluten && \
  bash ./dev/build-thirdparty.sh \
    --spark_version=${SPARK_VERSION} \
    2>&1 | tee ${TEE_FLAG} /opt/gluten/thirdparty.log
"

# ── Step 6: Verify JAR ↔ native lib consistency ──────────────────────────────
echo ""
echo "[post] Verifying JAR and native library consistency..."

JAR_PATH="${GLUTEN_DIR}/package/target/gluten-velox-bundle-spark${SPARK_VERSION}_2.12-linux_amd64-1.6.0-SNAPSHOT.jar"
CPP_LIBVELOX="${GLUTEN_DIR}/cpp/build/releases/libvelox.so"
CPP_LIBGLUTEN="${GLUTEN_DIR}/cpp/build/releases/libgluten.so"

SYNC_OK=true
if [ -f "$JAR_PATH" ] && [ -f "$CPP_LIBVELOX" ]; then
  JAR_MTIME=$(stat -c %Y "$JAR_PATH" 2>/dev/null || stat -f %m "$JAR_PATH" 2>/dev/null)
  SO_MTIME=$(stat -c %Y "$CPP_LIBVELOX" 2>/dev/null || stat -f %m "$CPP_LIBVELOX" 2>/dev/null)

  if [ "$SO_MTIME" -gt "$JAR_MTIME" ]; then
    echo " WARNING: libvelox.so ($(date -d @$SO_MTIME '+%H:%M:%S' 2>/dev/null || date -r $SO_MTIME '+%H:%M:%S')) is newer than JAR ($(date -d @$JAR_MTIME '+%H:%M:%S' 2>/dev/null || date -r $JAR_MTIME '+%H:%M:%S'))"
    echo "          JAR may contain stale native libraries!"
    SYNC_OK=false
  fi

  JAR_SO_MD5=$(unzip -p "$JAR_PATH" linux/amd64/libvelox.so 2>/dev/null | md5sum | awk '{print $1}')
  CPP_SO_MD5=$(md5sum "$CPP_LIBVELOX" | awk '{print $1}')
  if [ "$JAR_SO_MD5" != "$CPP_SO_MD5" ]; then
    echo " WARNING: libvelox.so MD5 mismatch!"
    echo "          JAR contains:  $JAR_SO_MD5"
    echo "          C++ produced:  $CPP_SO_MD5"
    echo "          Injecting latest native libs into JAR..."
    docker exec "$CONTAINER_NAME" bash -c "
      mkdir -p /tmp/_jar_fix/linux/amd64 && \
      cp /opt/gluten/cpp/build/releases/libgluten.so /opt/gluten/cpp/build/releases/libvelox.so /tmp/_jar_fix/linux/amd64/ && \
      cd /tmp/_jar_fix && \
      jar uf /opt/gluten/package/target/gluten-velox-bundle-spark${SPARK_VERSION}_2.12-linux_amd64-1.6.0-SNAPSHOT.jar linux/amd64/libgluten.so linux/amd64/libvelox.so && \
      rm -rf /tmp/_jar_fix
    "
    echo "          Native libs injected. New JAR MD5:"
    md5sum "$JAR_PATH"
    SYNC_OK=true
  fi
fi

if [ "$SYNC_OK" = true ]; then
  echo " OK: JAR and native libraries are in sync."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " BUILD COMPLETE"
echo "=============================================="
if [ -f "$JAR_PATH" ]; then
  echo " JAR: $JAR_PATH"
  echo " MD5: $(md5sum "$JAR_PATH" | awk '{print $1}')"
else
  echo " JAR: ${GLUTEN_DIR}/package/target/  (check for exact filename)"
fi
echo ""
echo " To run with Spark:"
echo "   spark-shell \\"
echo "     --conf spark.plugins=org.apache.gluten.GlutenPlugin \\"
echo "     --conf spark.driver.extraClassPath=${JAR_PATH} \\"
echo "     --conf spark.executor.extraClassPath=${JAR_PATH} \\"
echo "     --conf spark.memory.offHeap.enabled=true \\"
echo "     --conf spark.memory.offHeap.size=20g \\"
echo "     --conf spark.gluten.sql.columnar.cudf=true"
echo "=============================================="
