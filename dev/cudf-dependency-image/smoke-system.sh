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
# Exercise one SYSTEM-cuDF dependency carrier as a real downstream consumer in
# fresh in-container Gluten and Velox source copies. The protected/nightly smoke
# requires clean, exact Gluten and Velox source revisions, resolves the carrier
# tag to one immutable image ID, runs as root with HOME=/root and JDK 17,
# invokes the existing public buildbundle entrypoint without its setup script,
# enables S3 for compile/link validation, uses caller-selected HDFS build
# support with an OFF default, keeps installed-cuDF compatibility checking
# explicitly ON and mismatch rebuilding explicitly OFF, and builds production
# Velox targets plus the Spark 3.5 Maven reactor. It proves the Gluten
# configuration retained the reconciled
# SYSTEM-cuDF selection, resolves the two required Arrow static archives from
# one /usr/local library directory, resolves AWS SDK from /usr/local, produces
# libgluten.so with no unresolved shared dependencies, builds Gluten Arrow from
# the carrier's prepared 15.0.0-gluten artifacts, and produces exactly one
# readable Spark 3.5.5/Scala 2.12 bundle. This smoke needs no AWS credentials,
# live S3 service, Hadoop runtime, or live HDFS service and does not validate
# Spark/JNI runtime, publish the carrier, or qualify arbitrary cross-process
# build-tree reuse. Trusted callers may
# supply one standard Maven settings.xml file; the smoke mounts only that
# resolved file read-only at Maven's root settings path.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GLUTEN_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
IMAGE=""
VELOX_DIR=""
CUDA_ARCH=""
MAVEN_SETTINGS=""
ENABLE_HDFS=OFF
NUM_THREADS=$(nproc)

usage() {
  cat <<EOF
Usage: $(basename "$0") --image=TAG --velox_dir=PATH --cuda_arch=ARCH [options]

Run one fresh root/JDK17 SYSTEM-mode buildbundle smoke against a locally built
SYSTEM-cuDF dependency carrier. The public entrypoint resolves the carrier's
installed Arrow and AWS SDK closures with S3 compile/link support enabled and
caller-selected HDFS build support, builds Velox and Gluten C++, proves
libgluten.so links without missing shared dependencies, and builds one readable
Spark 3.5.5/Scala 2.12 bundle through the Maven reactor. It makes no live S3,
HDFS, JNI, or Spark runtime request. Build parallelism defaults to
${NUM_THREADS} threads.

Optional:
  --enable_hdfs=ON|OFF   Build with HDFS support (default: ${ENABLE_HDFS})
  --maven_settings=PATH  Caller-supplied Maven settings.xml, mounted read-only
  --num_threads=N        Positive build parallelism (default: ${NUM_THREADS})
EOF
}

require_clean_git_head() {
  local source_name=$1
  local source_dir=$2
  local source_head
  local source_status

  if ! git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: ${source_name} source is not a Git worktree: ${source_dir}" >&2
    return 1
  fi
  if ! source_head=$(git -C "$source_dir" rev-parse --verify 'HEAD^{commit}'); then
    echo "ERROR: could not resolve ${source_name} source HEAD: ${source_dir}" >&2
    return 1
  fi
  if [[ ! "$source_head" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ${source_name} source HEAD is not a full SHA: ${source_head}" >&2
    return 1
  fi
  if ! source_status=$(git -C "$source_dir" status --porcelain=v1 \
    --untracked-files=all --ignore-submodules=none); then
    echo "ERROR: could not inspect ${source_name} source status: ${source_dir}" >&2
    return 1
  fi
  if [ -n "$source_status" ]; then
    echo "ERROR: ${source_name} source must be clean: ${source_dir}" >&2
    printf '%s\n' "$source_status" >&2
    return 1
  fi
  printf '%s\n' "$source_head"
}

for arg in "$@"; do
  case "$arg" in
    --image=*) IMAGE="${arg#*=}" ;;
    --velox_dir=*|--velox-dir=*) VELOX_DIR="${arg#*=}" ;;
    --cuda_arch=*|--cuda-arch=*) CUDA_ARCH="${arg#*=}" ;;
    --enable_hdfs=*) ENABLE_HDFS="${arg#*=}" ;;
    --maven_settings=*) MAVEN_SETTINGS="${arg#*=}" ;;
    --num_threads=*|--num-threads=*) NUM_THREADS="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: ${arg}" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$IMAGE" ] || [ -z "$VELOX_DIR" ] || [ -z "$CUDA_ARCH" ]; then
  echo "ERROR: --image, --velox_dir, and --cuda_arch are required" >&2
  usage >&2
  exit 2
fi
if [ "$ENABLE_HDFS" != ON ] && [ "$ENABLE_HDFS" != OFF ]; then
  echo "ERROR: --enable_hdfs must be ON or OFF" >&2
  exit 2
fi
if [ ! -d "$VELOX_DIR" ]; then
  echo "ERROR: Velox source directory does not exist: ${VELOX_DIR}" >&2
  exit 2
fi
if [[ "$IMAGE" == -* || "$IMAGE" == *[[:space:]]* ]]; then
  echo "ERROR: --image must be a Docker image reference without whitespace" >&2
  exit 2
fi
if [[ ! "$CUDA_ARCH" =~ ^[0-9]+(-real|-virtual)?(,[0-9]+(-real|-virtual)?)*$ ]]; then
  echo "ERROR: --cuda_arch must be an explicit numeric architecture list" >&2
  exit 2
fi
if [[ ! "$NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --num_threads must be a positive integer" >&2
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
command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker is required" >&2; exit 2; }
command -v timeout >/dev/null || { echo "ERROR: timeout is required" >&2; exit 2; }

GLUTEN_HEAD=$(require_clean_git_head Gluten "$GLUTEN_DIR")
VELOX_HEAD=$(require_clean_git_head Velox "$VELOX_DIR")
SOURCE_ARCHIVE_DIR=$(mktemp -d)
trap 'rm -rf "$SOURCE_ARCHIVE_DIR"' EXIT
GLUTEN_ARCHIVE="$SOURCE_ARCHIVE_DIR/gluten.tar"
VELOX_ARCHIVE="$SOURCE_ARCHIVE_DIR/velox.tar"
if ! git -C "$GLUTEN_DIR" archive --format=tar "$GLUTEN_HEAD" \
    > "$GLUTEN_ARCHIVE"; then
  echo "ERROR: could not archive Gluten source HEAD: ${GLUTEN_HEAD}" >&2
  exit 2
fi
if ! git -C "$VELOX_DIR" archive --format=tar "$VELOX_HEAD" \
    > "$VELOX_ARCHIVE"; then
  echo "ERROR: could not archive Velox source HEAD: ${VELOX_HEAD}" >&2
  exit 2
fi
test -s "$GLUTEN_ARCHIVE"
test -s "$VELOX_ARCHIVE"
if ! IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE"); then
  echo "ERROR: could not resolve dependency carrier image: ${IMAGE}" >&2
  exit 2
fi
if [[ ! "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: dependency carrier did not resolve to one full image ID: ${IMAGE_ID}" >&2
  exit 2
fi

echo "Smoke Gluten source HEAD: ${GLUTEN_HEAD}"
echo "Smoke Velox source HEAD: ${VELOX_HEAD}"
echo "Smoke dependency carrier image ID: ${IMAGE} -> ${IMAGE_ID}"
echo "Smoke providers: carrier-base glibc/loader and JDK 17; host-injected NVIDIA driver"
echo "Smoke HDFS build mode: ${ENABLE_HDFS}"
if [ -n "$MAVEN_SETTINGS" ]; then
  echo "Smoke Maven settings: caller-supplied file mounted read-only"
fi

docker_run_mounts=(
  --mount "type=bind,src=${GLUTEN_ARCHIVE},dst=/source/gluten.tar,readonly"
  --mount "type=bind,src=${VELOX_ARCHIVE},dst=/source/velox.tar,readonly"
)
if [ -n "$MAVEN_SETTINGS" ]; then
  docker_run_mounts+=(
    --mount "type=bind,src=${MAVEN_SETTINGS},dst=/root/.m2/settings.xml,readonly"
  )
fi

timeout --signal=TERM --kill-after=2m 45m \
  docker run --rm --platform=linux/amd64 --user 0:0 --gpus all \
  "${docker_run_mounts[@]}" \
  --env HOME=/root \
  --env "SMOKE_CUDA_ARCH=${CUDA_ARCH}" \
  --env "SMOKE_ENABLE_HDFS=${ENABLE_HDFS}" \
  --env "SMOKE_NUM_THREADS=${NUM_THREADS}" \
  "$IMAGE_ID" bash -lc '
set -euo pipefail
mkdir -p /smoke/gluten /smoke/velox
tar -C /smoke/gluten -xf /source/gluten.tar
tar -C /smoke/velox -xf /source/velox.tar

if [ -f /opt/rh/gcc-toolset-14/enable ]; then
  source /opt/rh/gcc-toolset-14/enable
fi
test "$(id -u)" -eq 0
test "$HOME" = /root
test -x "$JAVA_HOME/bin/java"
test -x "$JAVA_HOME/bin/javac"
java_version=$("$JAVA_HOME/bin/java" -version 2>&1 | head -1)
javac_version=$("$JAVA_HOME/bin/javac" -version 2>&1)
grep -Eq "version \"17([.]|\")" <<< "$java_version"
grep -Eq "^javac 17([.]|$)" <<< "$javac_version"
echo "Smoke Java runtime: ${java_version}; ${javac_version}"
export INSTALL_PREFIX=/usr/local
export TARGETS="velox velox_cudf_exec"
export NUM_THREADS="${SMOKE_NUM_THREADS}"
export GLUTEN_BUNDLE_MAVEN_PROFILES=backends-velox,spark-3.5,java-17

cd /smoke/gluten
spark_profile=$(sed -n "/<id>spark-3[.]5<\\/id>/,/<\\/profile>/p" pom.xml)
spark_version_line=$(grep -m1 -F "<spark.version>" <<< "$spark_profile")
spark_version=${spark_version_line#*<spark.version>}
spark_version=${spark_version%</spark.version>*}
scala_version_line=$(grep -m1 -F "<scala.binary.version>" pom.xml)
scala_binary_version=${scala_version_line#*<scala.binary.version>}
scala_binary_version=${scala_binary_version%</scala.binary.version>*}
test "$spark_version" = 3.5.5
test "$scala_binary_version" = 2.12
echo "Maven consumer versions: Spark ${spark_version}; Scala ${scala_binary_version}"
# GPU injection can otherwise pair /usr/local/bin/curl with the older system
# libcurl. Bootstrap Maven with the matching /usr/local library first.
LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  ./build/mvn --version
./dev/buildbundle-veloxbe.sh \
  --run_setup_script=OFF \
  --build_arrow=OFF \
  --build_tests=OFF \
  --build_examples=OFF \
  --build_benchmarks=OFF \
  --build_velox_tests=OFF \
  --build_velox_benchmarks=OFF \
  --enable_gpu=ON \
  --enable_s3=ON \
  --enable_hdfs="${SMOKE_ENABLE_HDFS}" \
  --cudf_source=SYSTEM \
  --cudf_version_info=/usr/local/share/gluten/cudf-build-info \
  --cudf_compatibility_check=ON \
  --rebuild_if_mismatch=OFF \
  --velox_home=/smoke/velox \
  --cuda_arch="${SMOKE_CUDA_ARCH}" \
  --num_threads="${SMOKE_NUM_THREADS}" \
  --spark_version=3.5

velox_cache=/smoke/velox/_build/release/CMakeCache.txt
gluten_cache=/smoke/gluten/cpp/build/CMakeCache.txt
cache_value() {
  local cache=$1
  local key=$2
  sed -n "s/^${key}:[^=]*=//p" "$cache" | tail -1
}

test "$(cache_value "$velox_cache" cudf_SOURCE)" = SYSTEM
test "$(cache_value "$velox_cache" VELOX_ENABLE_S3)" = ON
test "$(cache_value "$velox_cache" VELOX_ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"
test "$(cache_value "$gluten_cache" cudf_SOURCE)" = SYSTEM
test "$(cache_value "$gluten_cache" ENABLE_S3)" = ON
test "$(cache_value "$gluten_cache" ENABLE_HDFS)" = "$SMOKE_ENABLE_HDFS"
test "$(cache_value "$gluten_cache" CUDF_COMPATIBILITY_RECONCILED)" = ON
echo "Native feature boundary: S3 compile/link=ON; HDFS build=${SMOKE_ENABLE_HDFS}; Hadoop/libhdfs runtime supplied downstream and not exercised"

arrow_path=$(cache_value "$gluten_cache" ARROW_LIB_arrow)
arrow_bundled_path=$(cache_value "$gluten_cache" \
  ARROW_LIB_arrow_bundled_dependencies)
if [ -z "$arrow_path" ] || [ -z "$arrow_bundled_path" ]; then
  echo "ERROR: Gluten CMake did not record both Arrow static archives" >&2
  exit 1
fi
test "$(basename "$arrow_path")" = libarrow.a
test "$(basename "$arrow_bundled_path")" = libarrow_bundled_dependencies.a
arrow_dir=$(dirname "$arrow_path")
test "$arrow_dir" = "$(dirname "$arrow_bundled_path")"
case "$arrow_dir" in
  /usr/local/lib|/usr/local/lib64) ;;
  *)
    echo "ERROR: Arrow static closure resolved outside /usr/local/lib*: ${arrow_dir}" >&2
    exit 1
    ;;
esac
test -f "$arrow_path"
test -f "$arrow_bundled_path"
echo "Installed Arrow static closure: ${arrow_path}, ${arrow_bundled_path}"

check_awssdk_resolution() {
  local cache=$1
  local consumer=$2
  local cache_key=AWSSDK_DIR
  local sdk_path
  local sdk_config
  sdk_path=$(cache_value "$cache" "$cache_key")
  if [ -z "$sdk_path" ]; then
    cache_key=AWSSDK_ROOT_DIR
    sdk_path=$(cache_value "$cache" "$cache_key")
  fi
  if [ -z "$sdk_path" ]; then
    echo "ERROR: ${consumer} CMake did not record an installed AWS SDK resolution" >&2
    exit 1
  fi
  case "$sdk_path" in
    /usr/local|/usr/local/*) ;;
    *)
      echo "ERROR: ${consumer} AWS SDK resolved outside /usr/local: ${cache_key}=${sdk_path}" >&2
      exit 1
      ;;
  esac
  if [ "$cache_key" = AWSSDK_DIR ]; then
    sdk_config="${sdk_path}/AWSSDKConfig.cmake"
  else
    sdk_config=$(find "$sdk_path" -type f -name AWSSDKConfig.cmake -print -quit)
  fi
  test -n "$sdk_config"
  test -f "$sdk_config"
  echo "Installed ${consumer} AWS SDK resolution: ${cache_key}=${sdk_path}"
}

check_awssdk_resolution "$velox_cache" Velox
check_awssdk_resolution "$gluten_cache" Gluten

gluten_library=/smoke/gluten/cpp/build/releases/libgluten.so
test -s "$gluten_library"
if ! gluten_ldd=$(ldd "$gluten_library" 2>&1); then
  echo "ERROR: ldd could not inspect ${gluten_library}" >&2
  echo "$gluten_ldd" >&2
  exit 1
fi
if grep -Fq "not found" <<< "$gluten_ldd"; then
  echo "ERROR: libgluten.so has unresolved shared dependencies" >&2
  echo "$gluten_ldd" >&2
  exit 1
fi
echo "Linked Gluten shared library: ${gluten_library}"

mapfile -t gluten_arrow_jars < <(
  find /smoke/gluten/gluten-arrow/target -maxdepth 1 -type f \
    -name "gluten-arrow-*-3.5.jar" -print
)
if [ "${#gluten_arrow_jars[@]}" -ne 1 ]; then
  echo "ERROR: expected one Spark 3.5 Gluten Arrow JAR, found ${#gluten_arrow_jars[@]}" >&2
  printf "%s\n" "${gluten_arrow_jars[@]}" >&2
  exit 1
fi
jar tf "${gluten_arrow_jars[0]}" >/dev/null
echo "Built Spark 3.5 Gluten Arrow module: ${gluten_arrow_jars[0]}"

mapfile -t bundles < <(
  find /smoke/gluten/package/target -maxdepth 1 -type f \
    -name "gluten-velox-bundle-spark3.5_2.12-*.jar" -print
)
if [ "${#bundles[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one Spark 3.5 Gluten bundle, found ${#bundles[@]}" >&2
  printf "%s\n" "${bundles[@]}" >&2
  exit 1
fi
jar tf "${bundles[0]}" >/dev/null
echo "Readable Spark 3.5.5/Scala 2.12 Gluten bundle: ${bundles[0]}"
'
