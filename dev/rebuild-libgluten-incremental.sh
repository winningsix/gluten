#!/bin/bash
# Incremental rebuild of libgluten.so for the MPP Direction-B development loop.
#
# Run this script INSIDE one of the gluten build containers (gluten_q2_run or
# gluten_q2_build) where /opt/gluten is bind-mounted. It does NOT touch Java/Maven
# state -- only the cpp/build CMake project. After it runs, the next mvn test
# will pick up the new libgluten.so via the linux/amd64/libgluten.so copy that
# lives under backends-velox/target.
#
# Usage:
#   docker exec -e LD_LIBRARY_PATH=... <container> bash /opt/gluten/dev/rebuild-libgluten-incremental.sh [--clean] [--jobs N]
#
# Environment expected:
#   /opt/gluten             (bind mount of code/spark-gluten)
#   /opt/velox              (bind mount of code/velox)
#   /opt/velox/_build/release/_deps/{cudf,rmm,kvikio,rapids_logger}-build  on LD_LIBRARY_PATH
#
# Exit codes:
#   0 = build succeeded, libgluten.so updated, expected JNI symbols present
#   2 = wrong directory (not inside container with /opt/gluten)
#   3 = cmake configure failed
#   4 = make/build failed
#   5 = built libgluten.so missing or didn't grow new mtime
#   6 = expected JNI symbol absent (sanity check failed)

set -u
set -o pipefail

CLEAN=0
JOBS=$(nproc 2>/dev/null || echo 16)
LOGFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)  CLEAN=1; shift ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --log)    LOGFILE="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d /opt/gluten/cpp ]]; then
  echo "fatal: /opt/gluten/cpp not found. Run inside a container that bind-mounts the source." >&2
  exit 2
fi

cd /opt/gluten/cpp

BUILD_DIR=/opt/gluten/cpp/build
OUT_LIB=$BUILD_DIR/releases/libgluten.so
TARGET_LIB=/opt/gluten/backends-velox/target/scala-2.12/classes/linux/amd64/libgluten.so

if [[ -n "$LOGFILE" ]]; then
  exec > >(tee "$LOGFILE") 2>&1
fi

echo "==> rebuild-libgluten-incremental.sh"
echo "    CLEAN=$CLEAN  JOBS=$JOBS  BUILD_DIR=$BUILD_DIR"
date

if [[ $CLEAN -eq 1 ]]; then
  echo "==> --clean given: removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

# Capture pre-build mtime so we can detect "build said success but didn't actually relink".
PRE_MTIME=""
if [[ -f "$OUT_LIB" ]]; then
  PRE_MTIME=$(stat -c '%Y' "$OUT_LIB")
fi

# Use existing CMakeCache if present; otherwise re-configure.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo "==> No CMakeCache.txt; running fresh cmake configure"
  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"
  # Pull preserved options from the buildbundle script's defaults.
  # If a fresh configure is needed, the user should run buildbundle-veloxbe.sh
  # the first time. This script does NOT set up dependencies.
  echo "fatal: $BUILD_DIR/CMakeCache.txt absent. Run dev/buildbundle-veloxbe.sh once before using this incremental script." >&2
  exit 3
fi

echo "==> cmake --build (incremental)  [-j $JOBS]"
cmake --build "$BUILD_DIR" -j "$JOBS"
RC=$?

if [[ $RC -ne 0 ]]; then
  echo "fatal: cmake --build failed with rc=$RC" >&2
  exit 4
fi

if [[ ! -f "$OUT_LIB" ]]; then
  echo "fatal: $OUT_LIB not produced" >&2
  exit 5
fi

POST_MTIME=$(stat -c '%Y' "$OUT_LIB")
if [[ -n "$PRE_MTIME" && "$PRE_MTIME" == "$POST_MTIME" ]]; then
  echo "warning: libgluten.so mtime unchanged ($POST_MTIME). Either the build was a no-op (no source changes) or relink was skipped."
else
  echo "==> libgluten.so updated  ($PRE_MTIME -> $POST_MTIME)"
fi

# Sanity-check expected JNI symbols (these are present in any working libgluten.so;
# if they're missing the build is half-broken). If we want to assert that newly-
# refactored symbols are present we can grow this list as the refactor settles.
echo "==> JNI symbol sanity check"
for sym in \
  Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeCreateMppQuery \
  Java_org_apache_gluten_vectorized_MppQueryJniWrapper_nativeStartMppQuery \
  ; do
  if ! nm -D "$OUT_LIB" 2>/dev/null | grep -q " T $sym$"; then
    echo "fatal: expected JNI symbol absent: $sym" >&2
    exit 6
  fi
done
echo "    ok: all expected JNI symbols exported"

# Mirror to the location loaded at JVM startup. The Maven build copies cpp/build
# output here during `package`, but we want the next `mvn test` (without a fresh
# package) to see the new .so too.
if [[ -d "$(dirname "$TARGET_LIB")" ]]; then
  cp -p "$OUT_LIB" "$TARGET_LIB"
  echo "==> mirrored to $TARGET_LIB"
fi

echo "==> rebuild-libgluten-incremental.sh: SUCCESS"
date
exit 0
