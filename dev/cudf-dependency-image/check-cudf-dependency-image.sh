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
# Validate a completed local carrier through the exact checker installed in
# that image. The dependency carrier intentionally stores its prepared Maven
# repository under /root/.m2, so the wrapper runs one disposable root container
# with HOME=/root. GPU access is required because UCX exposes cuda_copy and
# cuda_ipc only when the CUDA driver is visible. The entrypoint owns the source
# marker, native dependencies, Arrow C++ and Java content, CUDA UCX, semantic
# AWS compile/link, package-resolution, and forbidden-artifact rules; this
# wrapper neither duplicates those rules nor performs registry publication.

set -euo pipefail

IMAGE=""
EXPECTED_COMMIT=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --image=*) IMAGE="${1#*=}" ;;
    --image)
      [ "$#" -ge 2 ] || { echo "ERROR: --image requires a value" >&2; exit 2; }
      IMAGE=$2
      shift
      ;;
    --expected-cudf-commit=*) EXPECTED_COMMIT="${1#*=}" ;;
    --expected-cudf-commit)
      [ "$#" -ge 2 ] || { echo "ERROR: --expected-cudf-commit requires a value" >&2; exit 2; }
      EXPECTED_COMMIT=$2
      shift
      ;;
    *) echo "ERROR: unknown option: ${1}" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$IMAGE" ]; then
  echo "ERROR: --image is required" >&2
  exit 2
fi
if [[ "$IMAGE" == -* || "$IMAGE" == *[[:space:]]* ]]; then
  echo "ERROR: --image must be a Docker image reference without whitespace" >&2
  exit 2
fi
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: --expected-cudf-commit must be a full 40-character Git SHA" >&2
  exit 2
fi

docker run --rm --user 0:0 --gpus all \
  --env HOME=/root \
  --entrypoint /usr/local/bin/check-cudf-dependency-image-entrypoint.sh \
  "$IMAGE" \
  "$EXPECTED_COMMIT" \
  --require-cuda-transports=ON
