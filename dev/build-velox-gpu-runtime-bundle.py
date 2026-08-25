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

"""Turn completed Velox GPU build outputs into a relocatable deploy tree.

A normal Gluten build produces ``libgluten.so`` in a build directory where its
shared dependencies are resolved from the build container or installation
prefix. Copying that one file is not a deployable result: the dynamic loader
still needs its recursive ``DT_NEEDED`` closure, applications may open CUDA UCX
modules, RTCX CUDA JIT providers, and cuFile providers dynamically, consumers
load libraries through SONAME aliases, and build-time RPATHs may point back into
the build environment.

This source-owned packager closes that gap without rebuilding anything. Given
a real ``libgluten.so``, it discovers the libraries and CUDA UCX modules used
by the Velox GPU backend, adds the build-toolkit-matched NVRTC, NVRTC builtins,
nvJitLink, and libcufile families that are opened outside ``DT_NEEDED``,
preserves relative symlink families, and rewrites every copied ELF to use
bundle-relative ``$ORIGIN`` search paths. A previously assembled native tree
can instead be combined with one independently produced compatible Maven bundle
without repeating native discovery or relocation.

The command has three explicit combinations:

* ``--libgluten`` plus ``--native_only`` creates the reusable native tree.
* ``--libgluten`` plus ``--bundle_jar`` creates a same-run full bundle.
* ``--native_bundle`` plus ``--bundle_jar`` composes an existing native tree
  with exactly one source-supported Spark/Scala Maven artifact.

Every native tree records only exact Gluten and Velox revisions and the Velox
backend type. Full assembly compares those properties with the existing
``gluten-build-info.properties`` inside the JAR. Spark and Scala identity is
inferred from the canonical JAR and its metadata, never from redundant command
line version arguments.

When ``--runtime_bundle_output`` is supplied, full mode is invoked by
``dev/cudf-dependency-image/smoke-system.sh`` after the real
``buildbundle-veloxbe.sh`` build succeeds. The disposable
``dev/velox-gpu-runtime-bundle/runtime-smoke.py`` consumes only that full layout
to test one clean local Spark/JNI/native-GPU path. For decoupled production,
``dev/build-velox-jvm-bundle.py`` builds one Maven artifact and this packager's
``--native_bundle`` mode performs the identity-checked join. Native-only mode
remains an intermediate packaging boundary; the eventual runtime still owns
exact-distribution JNI and application qualification.

The packager must run where the completed ELF dependencies still resolve. It
does not compile Gluten or Velox, download missing libraries, create JVM
artifacts, or test the resulting application.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

METADATA_DIR = Path(__file__).resolve().parent / "velox-gpu-runtime-bundle"
sys.path.insert(0, str(METADATA_DIR))
from artifact_metadata import (  # noqa: E402
    ArtifactMetadataError,
    BundleJarInfo,
    CANONICAL_BUNDLE_JAR_RE,
    NATIVE_BUILD_INFO_NAME,
    NativeBuildInfo,
    native_build_info,
    read_bundle_jar,
    read_native_build_info,
    validate_jar_source_support,
    validate_native_jar_identity,
    write_native_build_info,
)

ROOT_POM = Path(__file__).resolve().parent.parent / "pom.xml"
NEEDED_RE = re.compile(r"\(NEEDED\).*Shared library: \[([^]]+)]")
SONAME_RE = re.compile(r"\(SONAME\).*Library soname: \[([^]]+)]")
UCX_GLOBS = ("/usr/local/lib*/ucx",)
CUDA_RUNTIME_SONAME_RE = re.compile(r"^libcudart[.]so[.]([0-9]+)$")
RTCX_FAMILY_STEMS = (
    "libnvrtc.so",
    "libnvrtc-builtins.so",
    "libnvJitLink.so",
)
CUFILE_FAMILY_STEM = "libcufile.so"
CUFILE_SONAME = "libcufile.so.0"

# These dependencies are deliberately supplied by the target runtime instead
# of being copied. Everything else named by DT_NEEDED must become bundle
# content. The path checks below keep this exception list from accepting an
# arbitrary library with a familiar basename.
GLIBC_PROVIDERS = {
    "ld-linux-aarch64.so.1",
    "ld-linux-x86-64.so.2",
    "ld64.so.1",
    "ld64.so.2",
    "libBrokenLocale.so.1",
    "libanl.so.1",
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libmemusage.so",
    "libnsl.so.1",
    "libpcprofile.so",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libthread_db.so.1",
    "libutil.so.1",
    "linux-vdso.so.1",
}

JDK17_PROVIDER_SONAMES = {
    "libattach.so",
    "libawt.so",
    "libawt_headless.so",
    "libdt_socket.so",
    "libextnet.so",
    "libfontmanager.so",
    "libinstrument.so",
    "libjava.so",
    "libjawt.so",
    "libjimage.so",
    "libjli.so",
    "libjsig.so",
    "libjsound.so",
    "libjvm.so",
    "libmanagement.so",
    "libmanagement_agent.so",
    "libmanagement_ext.so",
    "libnet.so",
    "libnio.so",
    "libprefs.so",
    "libsctp.so",
    "libsyslookup.so",
    "libverify.so",
    "libzip.so",
}


class PackagerError(RuntimeError):
    """Expected input or bundle-validation failure."""


@dataclass
class PlannedObject:
    source: Path
    group: str
    soname: str
    aliases: dict[str, str] = field(default_factory=dict)


def _run(command: Sequence[str], description: str, allow_failure: bool = False):
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise PackagerError(f"required tool not found: {command[0]}") from error
    if result.returncode != 0 and not allow_failure:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise PackagerError(f"{description} failed{suffix}")
    return result


def _preflight_tools(*, require_jar: bool):
    commands = ("ldd", "patchelf", "readelf")
    if require_jar:
        commands = ("jar", *commands)
    for command in commands:
        if shutil.which(command) is None:
            raise PackagerError(f"required tool not found: {command}")


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _normalized_child(parent: Path, target: str) -> Path:
    return Path(os.path.normpath(os.path.join(parent, target)))


def _resolve_source(path: Path, label: str) -> Path:
    """Resolve a source file, rejecting unsafe leaf symlink chains."""
    try:
        parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PackagerError(f"{label} has an inaccessible parent: {path}") from error

    current = parent / path.name
    seen: set[Path] = set()
    while current.is_symlink():
        if current in seen:
            raise PackagerError(f"{label} has a cyclic symlink: {path}")
        seen.add(current)
        try:
            target = os.readlink(current)
        except OSError as error:
            raise PackagerError(f"cannot read {label} symlink: {current}") from error
        if os.path.isabs(target):
            raise PackagerError(f"{label} has an absolute symlink: {current}")
        current = _normalized_child(current.parent, target)
        if not _is_within(current, parent):
            raise PackagerError(f"{label} symlink escapes its directory: {path}")

    if not current.exists():
        raise PackagerError(f"{label} has a broken symlink or is missing: {path}")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PackagerError(f"cannot resolve {label}: {path}") from error
    if not _is_within(resolved, parent):
        raise PackagerError(f"{label} symlink escapes its directory: {path}")
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise PackagerError(f"cannot stat {label}: {path}") from error
    if not stat.S_ISREG(mode) or not os.access(resolved, os.R_OK):
        raise PackagerError(f"{label} is not a readable regular file: {path}")
    return resolved


def _validate_jar(path: Path) -> BundleJarInfo:
    try:
        info = read_bundle_jar(path)
        validate_jar_source_support(info, ROOT_POM)
    except ArtifactMetadataError as error:
        raise PackagerError(str(error)) from error
    _run(["jar", "tf", str(info.path)], f"reading bundle JAR {path}")
    return info


def _validate_libgluten(path: Path) -> Path:
    if path.name != "libgluten.so":
        raise PackagerError(f"libgluten input must be named libgluten.so: {path}")
    if path.is_symlink():
        raise PackagerError(f"libgluten must be a real file, not a symlink: {path}")
    resolved = _resolve_source(path, "libgluten")
    if not _is_elf(resolved):
        raise PackagerError(f"libgluten is not an ELF file: {path}")
    return resolved


def _validate_output_dir(path: Path) -> Path:
    # Requiring an empty output prevents stale files from satisfying final
    # shape or dependency checks and appearing to belong to this invocation.
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise PackagerError(f"output directory does not exist: {path}") from error
    if not stat.S_ISDIR(mode):
        raise PackagerError(f"output path is not a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise PackagerError(f"output directory is not writable: {path}")
    try:
        if next(path.iterdir(), None) is not None:
            raise PackagerError(f"output directory must be empty: {path}")
    except OSError as error:
        raise PackagerError(f"cannot inspect output directory: {path}") from error
    return path.resolve(strict=True)


def _discover_ucx_dirs(patterns: Iterable[str] = UCX_GLOBS) -> list[Path]:
    directories: set[Path] = set()
    for pattern in patterns:
        for match in sorted(Path("/").glob(pattern.removeprefix("/"))):
            if match.is_dir():
                directories.add(match.resolve(strict=True))
    return sorted(directories)


def _ucx_entries(ucx_dirs: Iterable[Path]) -> tuple[list[Path], set[Path]]:
    """Collect UCX modules that DT_NEEDED traversal cannot discover.

    UCX selects transport and memory-hook modules dynamically, so they may not
    appear in ``libgluten.so``'s linker closure. Treat every installed
    ``*.so*`` module as a closure root and require both CUDA module families.
    """

    entries: list[Path] = []
    real_files: set[Path] = set()
    has_uct_cuda = False
    has_ucm_cuda = False

    directories = sorted(
        {Path(directory).resolve(strict=True) for directory in ucx_dirs}
    )
    for directory in directories:
        if not directory.is_dir():
            raise PackagerError(f"UCX module path is not a directory: {directory}")
        try:
            candidates = sorted(directory.glob("*.so*"))
        except OSError as error:
            raise PackagerError(
                f"cannot inspect UCX module directory: {directory}"
            ) from error
        for candidate in candidates:
            if not candidate.is_symlink() and not candidate.exists():
                continue
            resolved = _resolve_source(candidate, "UCX module")
            if not _is_elf(resolved):
                raise PackagerError(f"UCX module is not an ELF file: {candidate}")
            entries.append(candidate)
            real_files.add(resolved)
            has_uct_cuda = has_uct_cuda or candidate.name.startswith("libuct_cuda.so")
            has_ucm_cuda = has_ucm_cuda or candidate.name.startswith("libucm_cuda.so")

    if not has_uct_cuda:
        raise PackagerError("required CUDA UCX module libuct_cuda.so* was not found")
    if not has_ucm_cuda:
        raise PackagerError("required CUDA UCX module libucm_cuda.so* was not found")
    return entries, real_files


def _source_aliases(real: Path) -> dict[str, str]:
    """Return the relative symlink family that resolves to one real library."""

    aliases: dict[str, str] = {}
    try:
        candidates = sorted(real.parent.iterdir())
    except OSError as error:
        raise PackagerError(
            f"cannot inspect shared-library family: {real.parent}"
        ) from error

    for candidate in candidates:
        if ".so" not in candidate.name:
            continue
        if not candidate.is_symlink():
            continue
        try:
            resolved = _resolve_source(candidate, "shared-library alias")
        except PackagerError:
            continue
        if resolved != real:
            continue
        aliases[candidate.name] = os.readlink(candidate)
    return aliases


def _needed(path: Path) -> list[str]:
    result = _run(["readelf", "-d", str(path)], f"readelf on {path}")
    return [
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := NEEDED_RE.search(line))
    ]


def _soname(path: Path) -> str:
    result = _run(["readelf", "-d", str(path)], f"readelf on {path}")
    names = [
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := SONAME_RE.search(line))
    ]
    if len(names) > 1:
        raise PackagerError(f"shared library has multiple SONAME entries: {path}")
    return names[0] if names else path.name


def _shared_library_family(directory: Path, stem: str, label: str) -> tuple[Path, str]:
    """Resolve one coherent ``*.so*`` family from a single directory."""

    try:
        candidates = sorted(directory.glob(f"{stem}*"))
    except OSError as error:
        raise PackagerError(f"cannot inspect {label} directory: {directory}") from error
    if not candidates:
        raise PackagerError(f"required {label} family {stem}* was not found")

    real_files: set[Path] = set()
    for candidate in candidates:
        resolved = _resolve_source(candidate, label)
        if not _is_elf(resolved):
            raise PackagerError(f"{label} is not an ELF file: {candidate}")
        real_files.add(resolved)
    if len(real_files) != 1:
        raise PackagerError(
            f"required {label} family {stem}* resolves to multiple real files: "
            + ", ".join(str(path) for path in sorted(real_files))
        )

    real = next(iter(real_files))
    soname = _soname(real)
    available_names = {real.name, *_source_aliases(real)}
    if soname not in available_names:
        raise PackagerError(
            f"required {label} family does not provide its SONAME {soname}: {real}"
        )
    return real, soname


def _selected_cuda_runtime(
    objects: dict[Path, PlannedObject],
) -> tuple[Path, int]:
    """Return the one deployable CUDA runtime selected by libgluten's closure."""

    cuda_runtimes: list[tuple[Path, int]] = []
    for planned in objects.values():
        if match := CUDA_RUNTIME_SONAME_RE.fullmatch(planned.soname):
            cuda_runtimes.append((planned.source, int(match.group(1))))
    if len(cuda_runtimes) != 1:
        shown = ", ".join(f"{path} ({major})" for path, major in sorted(cuda_runtimes))
        suffix = f": {shown}" if shown else ""
        raise PackagerError(
            "libgluten closure must resolve exactly one versioned libcudart family"
            + suffix
        )

    cuda_runtime, cuda_major = cuda_runtimes[0]
    if "stubs" in cuda_runtime.parts:
        raise PackagerError(
            f"CUDA runtime resolved from a non-deployable stubs directory: {cuda_runtime}"
        )
    return cuda_runtime, cuda_major


def _rtcx_entries(objects: dict[Path, PlannedObject]) -> list[Path]:
    """Find RTCX dynamic providers beside the selected CUDA runtime.

    RTCX is statically absorbed into cuDF and opens these providers by name, so
    no ELF ``DT_NEEDED`` edge leads to them. The already resolved libcudart
    closure member supplies an unambiguous toolkit, CUDA-major, and architecture
    anchor; selecting a different installed toolkit would produce an incoherent
    runtime bundle.
    """

    cuda_runtime, cuda_major = _selected_cuda_runtime(objects)
    toolkit_lib = cuda_runtime.parent

    entries: list[Path] = []
    for stem in RTCX_FAMILY_STEMS:
        real, soname = _shared_library_family(toolkit_lib, stem, "RTCX provider")
        match = re.fullmatch(rf"{re.escape(stem)}[.]([0-9]+)(?:[.][0-9]+)*", soname)
        if match is None:
            raise PackagerError(
                f"RTCX provider has an unexpected versioned SONAME {soname}: {real}"
            )
        if int(match.group(1)) != cuda_major:
            raise PackagerError(
                f"RTCX provider {soname} does not match libcudart CUDA major "
                f"{cuda_major}: {real}"
            )
        entries.append(real)
    return entries


def _cufile_entries(objects: dict[Path, PlannedObject]) -> list[Path]:
    """Find the cuFile provider beside the selected CUDA runtime.

    KvikIO opens libcufile by name when compatibility mode is disabled, so no
    dependable ``DT_NEEDED`` edge from libgluten leads to it. Anchoring this
    explicit root beside the resolved libcudart keeps it in the same CUDA
    toolkit as the rest of the bundle.
    """

    cuda_runtime, _ = _selected_cuda_runtime(objects)
    toolkit_lib = cuda_runtime.parent

    stem = CUFILE_FAMILY_STEM
    real, soname = _shared_library_family(toolkit_lib, stem, "cuFile provider")
    if soname != CUFILE_SONAME:
        raise PackagerError(
            f"cuFile provider has SONAME {soname}, expected {CUFILE_SONAME}: {real}"
        )
    return [real]


def _parse_ldd_resolutions(output: str) -> dict[str, Path | None]:
    resolutions: dict[str, Path | None] = {}
    for line in output.splitlines():
        match = re.match(r"^\s*(\S+)\s+=>\s+(\S+)", line)
        if match:
            name, value = match.groups()
            resolutions[name] = None if value == "not" else Path(value)
            continue
        direct = re.match(r"^\s*(/\S+)\s+\(", line)
        if direct:
            path = Path(direct.group(1))
            resolutions[path.name] = path
    return resolutions


def _ldd_resolutions(path: Path) -> dict[str, Path | None]:
    result = _run(["ldd", str(path)], f"ldd on {path}", allow_failure=True)
    resolutions = _parse_ldd_resolutions(result.stdout + "\n" + result.stderr)
    if result.returncode != 0 and not resolutions:
        detail = result.stderr.strip() or result.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise PackagerError(f"ldd on {path} failed{suffix}")
    return resolutions


def _external_provider_kind(name: str) -> str | None:
    """Classify a dependency intentionally supplied by the target runtime."""

    if name in GLIBC_PROVIDERS:
        return "glibc"
    if re.fullmatch(r"libnss_[A-Za-z0-9_-]+\.so\.2", name):
        return "glibc"
    if name in JDK17_PROVIDER_SONAMES:
        return "jdk"
    if name in {"libcuda.so", "libcuda.so.1"}:
        return "driver"
    if name.startswith("libnvidia-") and ".so" in name:
        return "driver"
    if name.startswith("libnvcuvid.so"):
        return "driver"
    return None


def _external_provider_path_allowed(kind: str, resolution: Path | None) -> bool:
    """Validate provider paths, allowing runtime-injected driver/JDK libraries."""

    if resolution is None:
        return kind in {"driver", "jdk"}
    try:
        resolved = resolution.resolve(strict=True)
    except (OSError, RuntimeError):
        return False

    system_roots = [
        path.resolve(strict=True)
        for path in (Path("/lib"), Path("/lib64"), Path("/usr/lib"), Path("/usr/lib64"))
        if path.exists()
    ]
    if kind == "glibc":
        return any(_is_within(resolved, root) for root in system_roots)
    if kind == "driver":
        driver_roots = list(system_roots)
        for path in (Path("/usr/local/nvidia"), Path("/usr/local/cuda/compat")):
            if path.exists():
                driver_roots.append(path.resolve(strict=True))
        return any(_is_within(resolved, root) for root in driver_roots)
    if kind == "jdk":
        java_home_value = os.environ.get("JAVA_HOME")
        if not java_home_value:
            return False
        try:
            java_home = Path(java_home_value).resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return _is_within(resolved, java_home)
    return False


def _plan_objects(
    libgluten: Path, ucx_entries: Sequence[Path], ucx_reals: set[Path]
) -> dict[Path, PlannedObject]:
    """Plan one collision-free closure before writing any bundle content.

    ``readelf`` supplies the exact DT_NEEDED names while ``ldd`` maps those
    names to files in the current build environment. Planning reserves real
    basenames and every preserved alias up front so two different libraries
    cannot silently occupy the same destination name.

    The traversal starts with ``libgluten.so`` and drains its complete closure
    before adding the matching RTCX providers, cuFile providers, and then UCX
    modules. If multiple roots can resolve the same SONAME, this deterministic
    order keeps the object selected by the primary Gluten closure rather than
    replacing it with a later dynamically loaded root.
    """

    objects: dict[Path, PlannedObject] = {}
    namespace: dict[str, tuple[Path, str]] = {}
    queue: deque[Path] = deque()

    def reserve(name: str, real: Path, kind: str):
        previous = namespace.get(name)
        if previous is not None and previous[0] != real:
            raise PackagerError(
                f"shared-library basename collision for {name}: "
                f"{previous[0]} and {real}"
            )
        namespace[name] = (real, kind)

    def add(source: Path, requested_group: str) -> PlannedObject:
        real = _resolve_source(source, "shared library")
        if not _is_elf(real):
            raise PackagerError(f"shared-library closure member is not ELF: {source}")
        group = "ucx" if real in ucx_reals else requested_group
        if real in objects:
            existing = objects[real]
            if existing.group != group:
                raise PackagerError(
                    f"shared library would occupy two bundle locations: {real}"
                )
            return existing

        aliases = _source_aliases(real)
        soname = _soname(real)
        reserve(real.name, real, "real file")
        for alias in aliases:
            reserve(alias, real, "symlink")
        planned = PlannedObject(
            source=real, group=group, soname=soname, aliases=aliases
        )
        objects[real] = planned
        queue.append(real)
        return planned

    def drain_queue():
        while queue:
            source = queue.popleft()
            # readelf describes the loader contract; ldd identifies the file
            # currently satisfying each name. Both views are needed to copy
            # the right bytes while preserving the requested SONAME alias.
            resolutions = _ldd_resolutions(source)
            for needed_name in _needed(source):
                resolution = resolutions.get(needed_name)
                provider_kind = _external_provider_kind(needed_name)
                if provider_kind is not None:
                    if not _external_provider_path_allowed(provider_kind, resolution):
                        raise PackagerError(
                            f"external provider {needed_name} for {source} did not "
                            f"resolve from its allowed {provider_kind} location"
                        )
                    continue
                if resolution is None:
                    raise PackagerError(
                        f"unresolved dependency {needed_name} needed by {source}"
                    )
                selected = namespace.get(needed_name)
                if selected is None or objects[selected[0]].soname != needed_name:
                    dependency = add(resolution, "top")
                else:
                    # One process can load only one object for a SONAME. Keep
                    # the first resolution discovered from the ordered roots
                    # (the full libgluten closure, then the RTCX, cuFile, and
                    # UCX dynamic roots).
                    dependency = objects[selected[0]]
                available_names = {dependency.source.name, *dependency.aliases}
                if needed_name not in available_names:
                    raise PackagerError(
                        f"dependency {needed_name} has no preservable source alias "
                        f"for {source}"
                    )

    # Finish the primary closure first so its libcudart resolution selects the
    # one matching toolkit family used for the RTCX and cuFile providers.
    add(libgluten, "top")
    drain_queue()
    for entry in _rtcx_entries(objects):
        add(entry, "top")
        drain_queue()
    for entry in _cufile_entries(objects):
        add(entry, "top")
        drain_queue()

    # UCX modules are the final dynamic roots and retain their own subdirectory.
    for entry in ucx_entries:
        add(entry, "ucx")
        drain_queue()
    return objects


def _copy_and_patch(
    output: Path,
    bundle_jar: Path | None,
    objects: dict[Path, PlannedObject],
    build_info: NativeBuildInfo,
):
    """Materialize the plan, recreate aliases, and relocate ELF search paths."""

    libs = output / "libs"
    ucx = libs / "ucx"
    libs.mkdir()
    ucx.mkdir()
    write_native_build_info(libs / NATIVE_BUILD_INFO_NAME, build_info)
    if bundle_jar is not None:
        shutil.copy2(bundle_jar, output / bundle_jar.name)

    # Copy real files first. Recreating aliases only after all real targets
    # exist preserves the source SONAME chains without dereferencing them.
    destinations: dict[Path, Path] = {}
    for source, planned in sorted(objects.items(), key=lambda item: str(item[0])):
        directory = ucx if planned.group == "ucx" else libs
        destination = directory / source.name
        shutil.copy2(source, destination)
        destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
        destinations[source] = destination

    for source, planned in sorted(objects.items(), key=lambda item: str(item[0])):
        directory = ucx if planned.group == "ucx" else libs
        for alias, target in sorted(planned.aliases.items()):
            os.symlink(target, directory / alias)

    # Patch only real ELFs. Libraries in libs/ search beside themselves; UCX
    # modules also search the parent libs/ directory for their shared closure.
    for source, destination in sorted(
        destinations.items(), key=lambda item: str(item[0])
    ):
        rpath = "$ORIGIN:$ORIGIN/.." if objects[source].group == "ucx" else "$ORIGIN"
        _run(
            ["patchelf", "--set-rpath", rpath, str(destination)],
            f"patchelf on {destination}",
        )


def _validate_output_shape(output: Path, *, native_only: bool = False):
    """Enforce the exact public root layout for the selected content mode."""

    entries = list(output.iterdir())
    actual = {entry.name for entry in entries}
    if native_only:
        if actual != {"libs"}:
            raise PackagerError(
                "native-only bundle root must contain only libs/; "
                f"found {sorted(actual)}"
            )
        if not (output / "libs").is_dir() or not (output / "libs" / "ucx").is_dir():
            raise PackagerError("bundle must contain libs/ and libs/ucx/ directories")
        return

    jars = [
        entry
        for entry in entries
        if entry.is_file() and CANONICAL_BUNDLE_JAR_RE.fullmatch(entry.name)
    ]
    if len(jars) != 1:
        raise PackagerError(
            f"bundle must contain exactly one canonical Gluten bundle JAR; found {len(jars)}"
        )
    expected = {jars[0].name, "libs"}
    if actual != expected:
        raise PackagerError(
            f"bundle root has unexpected entries: {sorted(actual - expected)}"
        )
    if not (output / "libs").is_dir() or not (output / "libs" / "ucx").is_dir():
        raise PackagerError("bundle must contain libs/ and libs/ucx/ directories")
    _run(["jar", "tf", str(jars[0])], f"reading bundled JAR {jars[0]}")


def _validate_output_symlinks(output: Path):
    """Reject links that are absolute, broken, cyclic, or leave the bundle."""

    root = output.resolve(strict=True)
    for path in sorted(output.rglob("*")):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        if os.path.isabs(target):
            raise PackagerError(f"bundle contains an absolute symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PackagerError(
                f"bundle contains a broken or cyclic symlink: {path}"
            ) from error
        if not _is_within(resolved, root):
            raise PackagerError(f"bundle symlink escapes the bundle: {path}")
        if not resolved.is_file() or not _is_elf(resolved):
            raise PackagerError(
                f"bundle symlink does not resolve to a bundled ELF: {path}"
            )


def _validate_native_layout(output: Path) -> NativeBuildInfo:
    """Validate the only files and directories allowed in a native tree."""

    libs = output / "libs"
    ucx = libs / "ucx"
    if libs.is_symlink() or not libs.is_dir():
        raise PackagerError("bundle must contain the real directory libs/")
    if ucx.is_symlink() or not ucx.is_dir():
        raise PackagerError("bundle must contain the real directory libs/ucx/")

    info_path = libs / NATIVE_BUILD_INFO_NAME
    try:
        info = read_native_build_info(info_path)
    except ArtifactMetadataError as error:
        raise PackagerError(str(error)) from error

    for path in sorted(libs.rglob("*")):
        if path.is_symlink():
            continue
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise PackagerError(
                f"cannot inspect bundled native entry: {path}"
            ) from error
        if stat.S_ISDIR(mode):
            if path != ucx:
                raise PackagerError(f"bundle contains an unexpected directory: {path}")
            continue
        if not stat.S_ISREG(mode):
            raise PackagerError(f"bundle contains a special filesystem entry: {path}")
        if path == info_path:
            continue
        if not _is_elf(path):
            raise PackagerError(f"bundle contains an unexpected non-ELF file: {path}")

    libgluten = libs / "libgluten.so"
    if libgluten.is_symlink() or not libgluten.is_file() or not _is_elf(libgluten):
        raise PackagerError("bundle must contain a real ELF libs/libgluten.so")
    for stem in ("libuct_cuda.so", "libucm_cuda.so"):
        real, soname = _shared_library_family(ucx, stem, "CUDA UCX module")
        if real.parent.resolve(strict=True) != ucx.resolve(strict=True):
            raise PackagerError(
                f"required CUDA UCX family {stem}* resolves outside libs/ucx/: {real}"
            )
        if re.fullmatch(rf"{re.escape(stem)}[.][0-9]+(?:[.][0-9]+)*", soname) is None:
            raise PackagerError(
                f"required CUDA UCX family {stem}* has unexpected SONAME {soname}"
            )
    return info


def _validate_output_runpaths(output: Path):
    """Require the exact bundle-relative RUNPATH for every materialized ELF."""

    libs = output / "libs"
    ucx = libs / "ucx"
    for path in sorted(libs.rglob("*")):
        if path.is_symlink() or not path.is_file() or not _is_elf(path):
            continue
        expected = "$ORIGIN:$ORIGIN/.." if path.parent == ucx else "$ORIGIN"
        result = _run(
            ["patchelf", "--print-rpath", str(path)],
            f"reading bundled RUNPATH for {path}",
        )
        if result.stdout.strip() != expected:
            raise PackagerError(
                f"bundled ELF {path} has RUNPATH {result.stdout.strip()!r}; "
                f"expected {expected!r}"
            )


def _validate_output_rtcx(output: Path):
    """Recheck that the materialized CUDA and RTCX families remain coherent."""

    libs = output / "libs"
    _, cudart_soname = _shared_library_family(libs, "libcudart.so", "CUDA runtime")
    match = CUDA_RUNTIME_SONAME_RE.fullmatch(cudart_soname)
    if match is None:
        raise PackagerError(
            f"bundled CUDA runtime has an unexpected SONAME: {cudart_soname}"
        )
    cuda_major = int(match.group(1))
    for stem in RTCX_FAMILY_STEMS:
        real, soname = _shared_library_family(libs, stem, "RTCX provider")
        provider_match = re.fullmatch(
            rf"{re.escape(stem)}[.]([0-9]+)(?:[.][0-9]+)*", soname
        )
        if provider_match is None or int(provider_match.group(1)) != cuda_major:
            raise PackagerError(
                f"bundled RTCX provider {soname} does not match libcudart CUDA "
                f"major {cuda_major}: {real}"
            )


def _validate_output_cufile(output: Path):
    """Require the KvikIO cuFile provider family in the materialized bundle."""

    libs = output / "libs"
    stem = CUFILE_FAMILY_STEM
    real, soname = _shared_library_family(libs, stem, "cuFile provider")
    if soname != CUFILE_SONAME:
        raise PackagerError(
            f"bundled cuFile provider has SONAME {soname}, expected "
            f"{CUFILE_SONAME}: {real}"
        )


def _validate_materialized_bundle(
    output: Path, *, native_only: bool
) -> tuple[NativeBuildInfo, BundleJarInfo | None]:
    """Re-read a native or full tree after build, copy, or caller handoff."""

    _validate_output_shape(output, native_only=native_only)
    _validate_output_symlinks(output)
    native_info = _validate_native_layout(output)
    _validate_output_rtcx(output)
    _validate_output_cufile(output)
    _validate_output_runpaths(output)
    _validate_output_needed(output)
    if native_only:
        return native_info, None

    jars = [
        path
        for path in output.iterdir()
        if path.is_file() and CANONICAL_BUNDLE_JAR_RE.fullmatch(path.name)
    ]
    jar_info = _validate_jar(jars[0])
    try:
        validate_native_jar_identity(native_info, jar_info)
    except ArtifactMetadataError as error:
        raise PackagerError(str(error)) from error
    return native_info, jar_info


def _validate_output_needed(output: Path):
    """Check that every non-provider DT_NEEDED name resolves inside the bundle."""

    libs = output / "libs"
    ucx = libs / "ucx"
    real_elfs = [
        path
        for path in sorted(libs.rglob("*"))
        if path.is_file() and not path.is_symlink() and _is_elf(path)
    ]
    for elf in real_elfs:
        search_dirs = [ucx, libs] if elf.parent == ucx else [libs]
        for needed_name in _needed(elf):
            if _external_provider_kind(needed_name) is not None:
                continue
            candidates = [directory / needed_name for directory in search_dirs]
            match = next(
                (candidate for candidate in candidates if candidate.exists()), None
            )
            if match is None:
                raise PackagerError(
                    f"bundled ELF {elf} cannot resolve {needed_name} inside the bundle"
                )
            try:
                resolved = match.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise PackagerError(
                    f"bundled dependency {match} is broken or cyclic"
                ) from error
            if not _is_within(resolved, output.resolve(strict=True)) or not _is_elf(
                resolved
            ):
                raise PackagerError(
                    f"bundled dependency {match} does not resolve to a bundled ELF"
                )
            resolved_soname = _soname(resolved)
            if resolved_soname != needed_name:
                raise PackagerError(
                    f"bundled dependency {match} resolves to ELF SONAME "
                    f"{resolved_soname}, expected {needed_name}"
                )


def build_bundle(
    bundle_jar: Path | None,
    libgluten: Path,
    output_dir: Path,
    *,
    gluten_revision: str,
    velox_revision: str,
    ucx_dirs: Sequence[Path] | None = None,
) -> Path:
    """Build a full bundle or native closure; ``ucx_dirs`` is test-only input."""

    native_only = bundle_jar is None
    _preflight_tools(require_jar=not native_only)
    try:
        build_info = native_build_info(gluten_revision, velox_revision)
    except ArtifactMetadataError as error:
        raise PackagerError(str(error)) from error
    jar_info = None if bundle_jar is None else _validate_jar(Path(bundle_jar))
    if jar_info is not None:
        try:
            validate_native_jar_identity(build_info, jar_info)
        except ArtifactMetadataError as error:
            raise PackagerError(str(error)) from error
    gluten = _validate_libgluten(Path(libgluten))
    output = _validate_output_dir(Path(output_dir))
    directories = _discover_ucx_dirs() if ucx_dirs is None else list(ucx_dirs)
    entries, ucx_reals = _ucx_entries(directories)

    # Resolve the entire closure and all name collisions before modifying the
    # caller's empty output directory. Discovery failures therefore cannot
    # leave a bundle that merely looks complete.
    objects = _plan_objects(gluten, entries, ucx_reals)
    _copy_and_patch(
        output,
        None if jar_info is None else jar_info.path,
        objects,
        build_info,
    )

    # Re-read the produced tree as a consumer would: exact shape first, then
    # filesystem containment, and finally loader-name resolution.
    _validate_materialized_bundle(output, native_only=native_only)
    return output


def compose_bundle(native_bundle: Path, bundle_jar: Path, output_dir: Path) -> Path:
    """Copy one validated native tree and add one identity-matched JVM JAR."""

    _preflight_tools(require_jar=True)
    input_path = Path(native_bundle).expanduser()
    try:
        native = input_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PackagerError(f"native bundle does not resolve: {input_path}") from error
    if not native.is_dir():
        raise PackagerError(f"native bundle must name a directory: {input_path}")
    output = _validate_output_dir(Path(output_dir))
    if output == native or _is_within(output, native) or _is_within(native, output):
        raise PackagerError("native bundle and output directory must not overlap")

    native_info, _ = _validate_materialized_bundle(native, native_only=True)
    jar_info = _validate_jar(Path(bundle_jar))
    try:
        validate_native_jar_identity(native_info, jar_info)
    except ArtifactMetadataError as error:
        raise PackagerError(str(error)) from error

    # Composition deliberately copies the already-relocated tree as data. It
    # does not discover dependencies, resolve the build host, or patch ELFs.
    shutil.copytree(
        native / "libs",
        output / "libs",
        symlinks=True,
        copy_function=shutil.copy2,
    )
    shutil.copy2(jar_info.path, output / jar_info.path.name)
    _validate_materialized_bundle(output, native_only=False)
    return output


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Assemble a Spark-Gluten Velox GPU runtime bundle"
    )
    content = parser.add_mutually_exclusive_group(required=True)
    content.add_argument("--bundle_jar", type=Path)
    content.add_argument(
        "--native_only",
        action="store_true",
        help="assemble only libs/ and libs/ucx/ without a JVM bundle JAR",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--libgluten", type=Path)
    source.add_argument(
        "--native_bundle",
        type=Path,
        help="compose an existing native-only bundle with --bundle_jar",
    )
    parser.add_argument("--gluten_revision")
    parser.add_argument("--velox_revision")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.native_bundle is not None and args.native_only:
        parser.error("--native_bundle requires --bundle_jar")
    if args.libgluten is not None:
        if args.gluten_revision is None or args.velox_revision is None:
            parser.error("--libgluten requires --gluten_revision and --velox_revision")
    elif args.gluten_revision is not None or args.velox_revision is not None:
        parser.error("revision options are not accepted with --native_bundle")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.native_bundle is not None:
            output = compose_bundle(
                args.native_bundle, args.bundle_jar, args.output_dir
            )
        else:
            output = build_bundle(
                args.bundle_jar,
                args.libgluten,
                args.output_dir,
                gluten_revision=args.gluten_revision,
                velox_revision=args.velox_revision,
            )
    except PackagerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.native_only:
        print(f"Spark-Gluten Velox GPU native runtime closure assembled at {output}")
    else:
        print(f"Spark-Gluten Velox GPU runtime bundle assembled at {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
