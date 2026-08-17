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
# Keep the installed-cuDF SYSTEM build contract stable with a fast, hermetic
# premerge check. This script tests policy decisions and argument forwarding;
# it does not compile cuDF or Velox, or qualify CUDA, ABI, JNI, or Spark behavior.
#
# How it works:
# - Create temporary matching, mismatching, missing, malformed, and ambiguous
#   source-marker fixtures.
# - Drive the verifier and public build entrypoints with fake build tools, then
#   assert status classification, strict/skip/fallback behavior, and CMake flags.
# - Statically assert that the real CMake and dependency-image smoke paths
#   retain their required defaults and explicit strict SYSTEM settings.
# A real dependency carrier and null-mask executable are reserved for the
# protected-nightly smoke contract.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
GLUTEN_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
VERIFIER="$SCRIPT_DIR/verify-system-cudf.py"
ENTRYPOINT="$SCRIPT_DIR/builddeps-veloxbe.sh"
VELOX_ENTRYPOINT="$GLUTEN_DIR/ep/build-velox/src/build-velox.sh"
COMMIT=5beaa5954688fcb12236ffb434e192ea2c77db30
OTHER_COMMIT=33320d64c94a64c94bccc5e2c522721e4d275858

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
velox="$tmp/velox"
mkdir -p "$velox/CMake/resolve_dependency_modules"
cat > "$velox/CMake/resolve_dependency_modules/cudf.cmake" <<EOF
set(VELOX_cudf_COMMIT ${COMMIT})
EOF
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$COMMIT" > "$tmp/match"
printf 'CUDF_COMMIT=%s\nCUDF_VERSION=26.08\n' "$OTHER_COMMIT" > "$tmp/mismatch"
printf 'CUDF_COMMIT=short\n' > "$tmp/malformed"

expect_status() {
  local expected=$1
  local name=$2
  shift 2
  local status=0
  "$@" > "$tmp/${name}.out" 2> "$tmp/${name}.err" || status=$?
  if [ "$status" -ne "$expected" ]; then
    echo "ERROR: ${name}: expected status ${expected}, got ${status}" >&2
    cat "$tmp/${name}.out" "$tmp/${name}.err" >&2
    exit 1
  fi
}

expect_status 0 match "$VERIFIER" --velox-home "$velox" --version-info "$tmp/match"
grep -Fq 'SYSTEM cuDF compatibility verified' "$tmp/match.out"
expect_status 10 mismatch "$VERIFIER" --velox-home "$velox" --version-info "$tmp/mismatch"
grep -Fq 'installed cuDF does not match the selected Velox source' "$tmp/mismatch.err"
expect_status 10 malformed "$VERIFIER" --velox-home "$velox" --version-info "$tmp/malformed"
grep -Fq 'CUDF_COMMIT is not a full SHA' "$tmp/malformed.err"
expect_status 10 missing "$VERIFIER" --velox-home "$velox" --version-info "$tmp/missing"
grep -Fq 'action=reject-system-artifact' "$tmp/missing.err"

cp "$velox/CMake/resolve_dependency_modules/cudf.cmake" "$tmp/cudf.cmake"
printf 'set(VELOX_cudf_COMMIT short)\n' > "$velox/CMake/resolve_dependency_modules/cudf.cmake"
expect_status 11 expected_invalid "$VERIFIER" --velox-home "$velox" --version-info "$tmp/match"
grep -Fq 'VELOX_cudf_COMMIT is not a full SHA' "$tmp/expected_invalid.err"
mv "$tmp/cudf.cmake" "$velox/CMake/resolve_dependency_modules/cudf.cmake"
printf 'set(VELOX_cudf_COMMIT %s)\nset(VELOX_cudf_COMMIT %s)\n' \
  "$COMMIT" "$OTHER_COMMIT" > "$velox/CMake/resolve_dependency_modules/cudf.cmake"
expect_status 11 expected_ambiguous "$VERIFIER" --velox-home "$velox" \
  --version-info "$tmp/match"
grep -Fq 'found 2 VELOX_cudf_COMMIT declarations' "$tmp/expected_ambiguous.err"
printf 'set(VELOX_cudf_COMMIT %s)\n' "$COMMIT" \
  > "$velox/CMake/resolve_dependency_modules/cudf.cmake"

make_fake_tools() {
  local fake_bin=$1
  mkdir -p "$fake_bin"
  cat > "$fake_bin/uname" <<'EOF'
#!/bin/bash
if [ "${1:-}" = -s ]; then echo Linux; elif [ "${1:-}" = -m ]; then echo x86_64; else echo Linux; fi
EOF
  cat > "$fake_bin/nproc" <<'EOF'
#!/bin/bash
echo 4
EOF
  cat > "$fake_bin/make" <<'EOF'
#!/bin/bash
printf '<%s>\n' "$@" >> "${FAKE_MAKE_LOG:?}"
EOF
  chmod +x "$fake_bin/uname" "$fake_bin/nproc" "$fake_bin/make"
}

run_entrypoint() {
  local name=$1
  local expected=$2
  shift 2
  local fake_bin="$tmp/${name}-bin"
  local command=${SYSTEM_CUDF_TEST_COMMAND:-reconcile_cudf_source}
  make_fake_tools "$fake_bin"
  : > "$tmp/${name}.make"
  expect_status "$expected" "$name" env PATH="$fake_bin:$PATH" \
    FAKE_MAKE_LOG="$tmp/${name}.make" \
    "$ENTRYPOINT" --run_setup_script=OFF --build_arrow=OFF --enable_gpu=ON \
    --cudf_source=SYSTEM --velox_home="$velox" "$@" "$command"
}

run_entrypoint strict_fail 10 --cudf_version_info="$tmp/mismatch" \
  --cudf_compatibility_check=ON
grep -Fq 'action=fail' "$tmp/strict_fail.err"
test ! -s "$tmp/strict_fail.make"
cp "$velox/CMake/resolve_dependency_modules/cudf.cmake" "$tmp/cudf.cmake"
printf 'set(VELOX_cudf_COMMIT short)\n' > "$velox/CMake/resolve_dependency_modules/cudf.cmake"
run_entrypoint skip_no_read 0 --cudf_version_info="$tmp/missing" \
  --cudf_compatibility_check=OFF
grep -Fq 'compatibility check skipped' "$tmp/skip_no_read.err"
mv "$tmp/cudf.cmake" "$velox/CMake/resolve_dependency_modules/cudf.cmake"
SYSTEM_CUDF_TEST_COMMAND=build_velox run_entrypoint recognized_mismatch_fallback 0 \
  --cudf_version_info="$tmp/mismatch" --rebuild_if_mismatch=ON
grep -Fq 'action=rebuild-bundled' "$tmp/recognized_mismatch_fallback.out"
test -s "$tmp/recognized_mismatch_fallback.make"
grep -Fq -- '-Dcudf_SOURCE=BUNDLED' "$tmp/recognized_mismatch_fallback.make"
cp "$velox/CMake/resolve_dependency_modules/cudf.cmake" "$tmp/cudf.cmake"
printf 'set(VELOX_cudf_COMMIT short)\n' > "$velox/CMake/resolve_dependency_modules/cudf.cmake"
SYSTEM_CUDF_TEST_COMMAND=build_velox run_entrypoint invalid_expected_no_fallback 11 \
  --cudf_version_info="$tmp/match" --rebuild_if_mismatch=ON
test ! -s "$tmp/invalid_expected_no_fallback.make"
mv "$tmp/cudf.cmake" "$velox/CMake/resolve_dependency_modules/cudf.cmake"

run_velox_entrypoint() {
  local name=$1
  local expected=$2
  shift 2
  local fake_bin="$tmp/${name}-velox-bin"
  make_fake_tools "$fake_bin"
  : > "$tmp/${name}.make"
  expect_status "$expected" "$name" env PATH="$fake_bin:$PATH" \
    FAKE_MAKE_LOG="$tmp/${name}.make" \
    "$VELOX_ENTRYPOINT" --enable_gpu=ON --cudf_source=SYSTEM \
    --velox_home="$velox" --num_threads=1 "$@"
}

run_velox_entrypoint velox_match 0 --cudf_version_info="$tmp/match"
grep -Fq 'SYSTEM cuDF compatibility verified' "$tmp/velox_match.out"
test -s "$tmp/velox_match.make"
grep -Fq -- '-Dcudf_SOURCE=SYSTEM' "$tmp/velox_match.make"
run_velox_entrypoint velox_default_strict_fail 10 \
  --cudf_version_info="$tmp/mismatch"
test ! -s "$tmp/velox_default_strict_fail.make"

grep -Fq 'CUDF_COMPATIBILITY_CHECK=ON' "$ENTRYPOINT"
grep -Fq 'CUDF_COMPATIBILITY_CHECK=ON' "$VELOX_ENTRYPOINT"
grep -Fq 'option(CUDF_COMPATIBILITY_CHECK "Verify installed cuDF against Velox" ON)' \
  "$GLUTEN_DIR/cpp/CMakeLists.txt"
grep -Fq 'list(APPEND CUDF_VERIFY_OPTIONS --version-info "${CUDF_VERSION_INFO}")' \
  "$GLUTEN_DIR/cpp/CMakeLists.txt"
grep -Fq 'list(APPEND CUDF_VERIFY_OPTIONS --skip)' \
  "$GLUTEN_DIR/cpp/CMakeLists.txt"
grep -Fq 'if(NOT CUDF_VERIFY_RESULT EQUAL 0)' \
  "$GLUTEN_DIR/cpp/CMakeLists.txt"
grep -Fq -- '--cudf_compatibility_check=ON' \
  "$SCRIPT_DIR/cudf-dependency-image/smoke-system.sh"
grep -Fq -- '--rebuild_if_mismatch=OFF' \
  "$SCRIPT_DIR/cudf-dependency-image/smoke-system.sh"

if grep -ERn 'M[123] (source-identity marker|marker|verifier|parser)' \
  "$GLUTEN_DIR/docs/get-started/build-guide.md" \
  "$SCRIPT_DIR/build-cudf-dependency-image.sh" \
  "$SCRIPT_DIR/cudf-dependency-image"; then
  echo "ERROR: user-facing marker contract contains a development milestone name" >&2
  exit 1
fi

echo "SYSTEM cuDF compatibility policy tests passed"
