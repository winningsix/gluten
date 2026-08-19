#!/bin/bash

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
#

GLUTEN_ROOT=$(cd $(dirname -- $0)/..; pwd -P)

EXTRA_RESOURCE_DIR=$GLUTEN_ROOT/gluten-core/target/generated-resources
BUILD_INFO="$EXTRA_RESOURCE_DIR"/gluten-build-info.properties

# Delete old build-info file before regenerating
rm -f "$BUILD_INFO"
mkdir -p "$EXTRA_RESOURCE_DIR"

function echo_revision_info() {
  local revision=${GLUTEN_BUILD_INFO_REVISION:-}
  local branch=""
  local revision_time=""
  local remote_url=""
  if [ -z "$revision" ]; then
    revision=$(git -C "$GLUTEN_ROOT" rev-parse HEAD) || exit 1
    branch=$(git -C "$GLUTEN_ROOT" rev-parse --abbrev-ref HEAD) || exit 1
    revision_time=$(git -C "$GLUTEN_ROOT" show -s --format=%ci "$revision") || exit 1
    remote_url=$(git -C "$GLUTEN_ROOT" config --get remote.origin.url || true)
  fi
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: Gluten build-info revision must be one full lowercase Git SHA" >&2
    exit 1
  fi
  echo branch="$branch"
  echo revision="$revision"
  echo revision_time="$revision_time"
  echo date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  # Never embed credentials from a developer's local Git configuration in a
  # distributable JAR.  Strip URL userinfo while preserving the repository URL.
  remote_url=$(
    printf '%s' "$remote_url" |
      sed -E 's#^(https?://)[^/@]+(:[^/@]*)?@#\1#; s#[?#].*$##'
  )
  printf 'url=%s\n' "$remote_url"
}

function echo_velox_revision_info() {
  local backend_home=$1
  local revision=${GLUTEN_BUILD_INFO_VELOX_REVISION:-}
  local branch=""
  local revision_time=""
  if [ -z "$revision" ]; then
    revision=$(git -C "$backend_home" rev-parse HEAD) || exit 1
    branch=$(git -C "$backend_home" rev-parse --abbrev-ref HEAD) || exit 1
    revision_time=$(git -C "$backend_home" show -s --format=%ci "$revision") || exit 1
  fi
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: Velox build-info revision must be one full lowercase Git SHA" >&2
    exit 1
  fi
  local libgluten="$GLUTEN_ROOT/cpp/build/releases/libgluten.so"
  local gcc_version=""
  if [ -r "$libgluten" ]; then
    gcc_version=$(strings "$libgluten" | grep "GCC:" | head -n 1)
  fi
  echo gcc_version="$gcc_version"
  echo velox_branch="$branch"
  echo velox_revision="$revision"
  echo velox_revision_time="$revision_time"
}

function echo_clickhouse_revision_info() {
  echo ch_org=$(cat $GLUTEN_ROOT/cpp-ch/clickhouse.version | grep -oP '(?<=^CH_ORG=).*')
  echo ch_branch=$(cat $GLUTEN_ROOT/cpp-ch/clickhouse.version | grep -oP '(?<=^CH_BRANCH=).*')
  echo ch_commit=$(cat $GLUTEN_ROOT/cpp-ch/clickhouse.version | grep -oP '(?<=^CH_COMMIT=).*')
}

while (( "$#" )); do
  echo "$1"
  case $1 in
    --version)
      echo gluten_version="$2" >> "$BUILD_INFO"
      ;;
    --backend)
      BACKEND_TYPE="$2"
      echo backend_type="$BACKEND_TYPE" >> "$BUILD_INFO"
      # Compute backend home path based on type
      if [ "velox" = "$BACKEND_TYPE" ]; then
        BACKEND_HOME=${VELOX_HOME:-"$GLUTEN_ROOT/ep/build-velox/build/velox_ep"}
        echo_velox_revision_info "$BACKEND_HOME" >> "$BUILD_INFO"
      elif [ "ch" = "$BACKEND_TYPE" ] || [ "clickhouse" = "$BACKEND_TYPE" ]; then
        echo_clickhouse_revision_info >> "$BUILD_INFO"
      fi
      ;;
    --java)
      echo java_version="$2" >> "$BUILD_INFO"
      ;;
    --scala)
      echo scala_version="$2" >> "$BUILD_INFO"
      ;;
    --spark)
      echo spark_version="$2" >> "$BUILD_INFO"
      ;;
    --hadoop)
      echo hadoop_version="$2" >> "$BUILD_INFO"
      ;;
    --revision)
      if [ "true" = "$2" ]; then
        echo_revision_info >> "$BUILD_INFO"
      fi
      ;;
    *)
      echo "Error: $1 is not supported"
      ;;
  esac
  shift 2
done
