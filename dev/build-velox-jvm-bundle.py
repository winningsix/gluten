#!/usr/bin/env python3

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

"""Build one source-supported Velox JVM bundle without rebuilding native code.

The native runtime closure is independent of Spark and Scala, while Gluten's
JVM bundle is compiled for one source-owned Spark shim and Scala pairing.  This
entrypoint runs only the existing Maven reactor, stages exactly one canonical
bundle JAR, and verifies its embedded source identity so it can later be joined
with a matching native-only runtime bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Callable, Mapping, Sequence

# The entrypoint verifies a clean source tree after importing the shared helper.
# Avoid making that verification fail on bytecode created by this invocation.
sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
GLUTEN_ROOT = SCRIPT_DIR.parent
METADATA_DIR = SCRIPT_DIR / "velox-gpu-runtime-bundle"
if str(METADATA_DIR) not in sys.path:
    sys.path.insert(0, str(METADATA_DIR))

from artifact_metadata import (  # noqa: E402
    ArtifactMetadataError,
    BundleJarInfo,
    exact_spark_line,
    native_build_info,
    read_bundle_jar,
    select_source_profiles,
    validate_jar_source_support,
    validate_native_jar_identity,
)

FULL_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
Runner = Callable[..., subprocess.CompletedProcess[str]]


class JvmBundleBuildError(RuntimeError):
    """An input, Maven build, or produced-JAR validation failed."""


@dataclass(frozen=True)
class BuildResult:
    jar: Path
    gluten_revision: str
    velox_revision: str
    spark_version: str
    scala_binary: str


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=None if environment is None else dict(environment),
            check=False,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
        )
    except OSError as error:
        raise JvmBundleBuildError(f"could not run {command[0]}: {error}") from error
    if result.returncode != 0:
        detail = ""
        if capture_output:
            detail = "\n".join(
                text.strip() for text in (result.stdout, result.stderr) if text.strip()
            )
        suffix = f": {detail}" if detail else ""
        raise JvmBundleBuildError(f"command failed: {' '.join(command)}{suffix}")
    return result


def _existing_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise JvmBundleBuildError(f"{label} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise JvmBundleBuildError(f"{label} does not exist: {path}") from error
    if not stat.S_ISDIR(mode):
        raise JvmBundleBuildError(f"{label} is not a directory: {path}")
    return path.resolve(strict=True)


def _empty_output_directory(path: Path) -> Path:
    output = _existing_directory(path, "output directory")
    if not os.access(output, os.W_OK | os.X_OK):
        raise JvmBundleBuildError(f"output directory is not writable: {output}")
    try:
        first = next(output.iterdir(), None)
    except OSError as error:
        raise JvmBundleBuildError(
            f"could not inspect output directory: {output}"
        ) from error
    if first is not None:
        raise JvmBundleBuildError(f"output directory must be empty: {output}")
    return output


def _clean_git_head(path: Path, label: str, runner: Runner) -> str:
    source = _existing_directory(path, f"{label} source")
    # Containerized callers commonly bind-mount sources owned by the host user
    # while this build runs as root. Trust only this resolved source for each
    # read-only probe; do not mutate the caller's global Git configuration.
    git = ["git", "-c", f"safe.directory={source}", "-C", str(source)]
    inside = runner(
        [*git, "rev-parse", "--is-inside-work-tree"],
        cwd=source,
        environment=None,
    ).stdout.strip()
    if inside != "true":
        raise JvmBundleBuildError(f"{label} source is not a Git worktree: {source}")
    head = runner(
        [*git, "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=source,
        environment=None,
    ).stdout.strip()
    if FULL_REVISION_RE.fullmatch(head) is None:
        raise JvmBundleBuildError(f"{label} HEAD is not one full Git revision: {head}")
    status = runner(
        [
            *git,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=source,
        environment=None,
    ).stdout
    if status.strip():
        raise JvmBundleBuildError(f"{label} source must be clean: {source}")
    return head


def _require_jdk17(java_home: Path, runner: Runner, cwd: Path) -> Path:
    try:
        home = java_home.resolve(strict=True)
    except OSError as error:
        raise JvmBundleBuildError(f"JAVA_HOME does not exist: {java_home}") from error
    if not home.is_dir():
        raise JvmBundleBuildError(f"JAVA_HOME is not a directory: {java_home}")
    java = home / "bin" / "java"
    javac = home / "bin" / "javac"
    if not java.is_file() or not os.access(java, os.X_OK):
        raise JvmBundleBuildError(f"JDK java executable is missing: {java}")
    if not javac.is_file() or not os.access(javac, os.X_OK):
        raise JvmBundleBuildError(f"JDK javac executable is missing: {javac}")
    java_result = runner(
        [str(java), "-XshowSettings:properties", "-version"],
        cwd=cwd,
        environment=None,
    )
    settings = java_result.stdout + "\n" + java_result.stderr
    match = re.search(
        r"^\s*java[.]specification[.]version\s*=\s*([^\s]+)\s*$", settings, re.M
    )
    if match is None or match.group(1) != "17":
        found = match.group(1) if match is not None else "unknown"
        raise JvmBundleBuildError(
            f"Maven-only bundle build requires JDK 17, found {found}"
        )
    javac_result = runner([str(javac), "-version"], cwd=cwd, environment=None)
    javac_text = (javac_result.stdout + "\n" + javac_result.stderr).strip()
    if re.search(r"(?:^|\s)javac 17(?:[.]|\s|$)", javac_text) is None:
        raise JvmBundleBuildError(
            f"Maven-only bundle build requires javac 17, found {javac_text or 'unknown'}"
        )
    return home


def _matching_jar(
    target: Path,
    *,
    spark_line: str,
    scala_binary: str,
    spark_version: str,
    gluten_revision: str,
    velox_revision: str,
    pom: Path,
) -> BundleJarInfo:
    pattern = f"gluten-velox-bundle-spark{spark_line}_{scala_binary}-*.jar"
    candidates = sorted(target.glob(pattern)) if target.is_dir() else []
    expected_native = native_build_info(gluten_revision, velox_revision)
    matches: list[BundleJarInfo] = []
    rejected: list[str] = []
    for candidate in candidates:
        try:
            info = read_bundle_jar(candidate)
            validate_jar_source_support(info, pom)
            validate_native_jar_identity(expected_native, info)
            if info.spark_version != spark_version:
                raise ArtifactMetadataError(
                    f"embedded spark_version is {info.spark_version}, expected {spark_version}"
                )
            if info.scala_binary != scala_binary:
                raise ArtifactMetadataError(
                    f"embedded Scala binary is {info.scala_binary}, expected {scala_binary}"
                )
        except ArtifactMetadataError as error:
            rejected.append(f"{candidate.name}: {error}")
            continue
        matches.append(info)
    if len(matches) != 1:
        detail = "; ".join(rejected)
        suffix = f"; rejected: {detail}" if detail else ""
        raise JvmBundleBuildError(
            f"Maven must produce exactly one matching readable bundle JAR; "
            f"found {len(matches)}{suffix}"
        )
    return matches[0]


def build_jvm_bundle(
    *,
    spark_profile: str,
    scala_profile: str,
    velox_home: Path,
    output_dir: Path,
    spark_version: str | None = None,
    gluten_root: Path = GLUTEN_ROOT,
    java_home: Path | None = None,
    runner: Runner = _run_command,
) -> BuildResult:
    """Run only Maven and stage one identity-checked canonical bundle JAR."""

    root = _existing_directory(gluten_root, "Gluten source")
    velox = _existing_directory(velox_home, "Velox source")
    output = _empty_output_directory(output_dir)
    pom = root / "pom.xml"
    try:
        selected = select_source_profiles(pom, spark_profile, scala_profile)
    except ArtifactMetadataError as error:
        raise JvmBundleBuildError(str(error)) from error
    effective_spark = spark_version or selected.default_spark_version
    try:
        requested_line = exact_spark_line(effective_spark)
    except ArtifactMetadataError as error:
        raise JvmBundleBuildError(str(error)) from error
    if requested_line != selected.spark_line:
        raise JvmBundleBuildError(
            f"spark_version {effective_spark} is outside selected shim "
            f"{selected.spark_line}"
        )

    environment_java_home = java_home
    if environment_java_home is None:
        value = os.environ.get("JAVA_HOME", "")
        if not value:
            raise JvmBundleBuildError("JAVA_HOME must select a JDK 17 installation")
        environment_java_home = Path(value)
    environment_java_home = _require_jdk17(environment_java_home, runner, root)

    gluten_revision = _clean_git_head(root, "Gluten", runner)
    velox_revision = _clean_git_head(velox, "Velox", runner)
    maven = root / "build" / "mvn"
    if not maven.is_file() or not os.access(maven, os.X_OK):
        raise JvmBundleBuildError(f"Gluten Maven wrapper is missing: {maven}")

    profiles = f"backends-velox,{spark_profile},{scala_profile},java-17"
    command = [
        str(maven),
        "-pl",
        "package",
        "-am",
        "clean",
        "package",
        f"-P{profiles}",
        "-DskipTests",
        "-Dspotless.check.skip=true",
        f"-Dspark.version={effective_spark}",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "JAVA_HOME": str(environment_java_home),
            "VELOX_HOME": str(velox),
            "GLUTEN_BUILD_INFO_REVISION": gluten_revision,
            "GLUTEN_BUILD_INFO_VELOX_REVISION": velox_revision,
        }
    )
    # Maven can run for many minutes. Stream its reactor output so progress and
    # the actual failure are visible instead of buffering the entire build.
    runner(command, cwd=root, environment=environment, capture_output=False)

    info = _matching_jar(
        root / "package" / "target",
        spark_line=selected.spark_line,
        scala_binary=selected.scala_binary,
        spark_version=effective_spark,
        gluten_revision=gluten_revision,
        velox_revision=velox_revision,
        pom=pom,
    )
    destination = output / info.path.name
    shutil.copy2(info.path, destination)
    try:
        staged = read_bundle_jar(destination)
        validate_jar_source_support(staged, pom)
        validate_native_jar_identity(
            native_build_info(gluten_revision, velox_revision), staged
        )
    except ArtifactMetadataError as error:
        raise JvmBundleBuildError(f"staged bundle JAR is invalid: {error}") from error
    entries = list(output.iterdir())
    if entries != [destination]:
        raise JvmBundleBuildError(
            "Maven-only output must contain exactly one bundle JAR"
        )
    return BuildResult(
        jar=destination,
        gluten_revision=gluten_revision,
        velox_revision=velox_revision,
        spark_version=effective_spark,
        scala_binary=selected.scala_binary,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one Spark/Scala Gluten Velox JVM bundle with Maven only"
    )
    parser.add_argument("--spark_profile", required=True)
    parser.add_argument("--scala_profile", required=True)
    parser.add_argument("--velox_home", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--spark_version")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = build_jvm_bundle(
            spark_profile=args.spark_profile,
            scala_profile=args.scala_profile,
            velox_home=args.velox_home,
            output_dir=args.output_dir,
            spark_version=args.spark_version,
        )
    except JvmBundleBuildError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Maven-only Gluten source HEAD: {result.gluten_revision}")
    print(f"Maven-only Velox source HEAD: {result.velox_revision}")
    print(
        f"Maven-only JVM bundle: Spark {result.spark_version}; "
        f"Scala {result.scala_binary}; JDK 17; {result.jar}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
