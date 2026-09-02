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

"""Validate and execute one full Spark-Gluten Velox GPU runtime bundle.

The outer process validates the bundle and starts a disposable container; the
inner process rechecks its clean-runtime ELF closure and bundled RTCX providers
before launching the fixed Spark/JNI/cuDF query. Native-only deploy trees are
intentionally not accepted.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from artifact_metadata import (
    ArtifactMetadataError,
    BundleJarInfo,
    NATIVE_BUILD_INFO_NAME,
    read_bundle_jar,
    read_native_build_info,
    validate_native_jar_identity,
)
from host_platform import GLIBC_LOADERS, docker_platform_flag

SCRIPT_DIR = Path(__file__).resolve().parent
DOCKERFILE = SCRIPT_DIR / "runtime-smoke.Dockerfile"
BUNDLE_MOUNT = Path("/opt/gluten-deploy")
CALLER_PROPERTIES_MOUNT = Path("/opt/gluten-runtime-smoke/caller-spark-defaults.conf")
QUERY_SCRIPT = Path("/opt/gluten-runtime-smoke/runtime-query.py")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
ADAPTED_CUDF_OPERATOR_RE = re.compile(
    r"\bOperator: ID [0-9]+: " r"(CudfReduce(?:PARTIAL|FINAL|INTERMEDIATE|SINGLE))\["
)

PLUGIN_CLASS = "org.apache.gluten.GlutenPlugin"
SHUFFLE_MANAGER_CLASS = "org.apache.spark.shuffle.sort.ColumnarShuffleManager"
CUDA_RUNTIME_STEM = "libcudart.so"
RTCX_PROVIDER_STEMS = (
    "libnvrtc.so",
    "libnvrtc-builtins.so",
    "libnvJitLink.so",
)
NVRTC_PROBE_SOURCE = b"""\
extern "C" __global__ void rtcx_probe(float* values) {
  values[0] = __sinf(values[0]);
}
"""


class SmokeError(RuntimeError):
    """A concise, expected smoke failure."""


@dataclass(frozen=True)
class LibraryFamily:
    soname_path: Path
    real: Path
    version: tuple[int, ...]


@dataclass(frozen=True)
class RtcxProviders:
    cuda_major: int
    cudart: LibraryFamily
    nvrtc: LibraryFamily
    builtins: LibraryFamily
    nvjitlink: LibraryFamily


@dataclass(frozen=True)
class QualificationTarget:
    spark_version: str
    scala_binary: str
    java_version: str
    image: str


QUALIFICATION_TARGETS = {
    ("3.5.5", "2.12", "17"): QualificationTarget(
        spark_version="3.5.5",
        scala_binary="2.12",
        java_version="17",
        image="apache/spark:3.5.5-scala2.12-java17-python3-ubuntu",
    ),
    ("4.0.2", "2.13", "17"): QualificationTarget(
        spark_version="4.0.2",
        scala_binary="2.13",
        java_version="17",
        image="apache/spark:4.0.2-scala2.13-java17-python3-ubuntu",
    ),
}


@dataclass(frozen=True)
class Bundle:
    root: Path
    jar: Path
    jar_info: BundleJarInfo
    libgluten: Path
    ucx: Path
    rtcx: RtcxProviders


def _has_read_bit(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH))


def resolve_regular_file(
    value: str | os.PathLike[str], description: str, *, nonempty: bool
) -> Path:
    input_path = Path(value).expanduser()
    try:
        resolved = input_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SmokeError(f"{description} does not resolve: {input_path}") from error
    if not resolved.is_file() or not _has_read_bit(resolved):
        raise SmokeError(f"{description} must be a readable regular file: {input_path}")
    if nonempty and resolved.stat().st_size == 0:
        raise SmokeError(f"{description} must be nonempty: {input_path}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _walk_entries(root: Path) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted(dirnames + filenames):
            yield base / name


def validate_bundle_symlinks(root: Path) -> None:
    for entry in _walk_entries(root):
        if not entry.is_symlink():
            continue
        raw_target = os.readlink(entry)
        if os.path.isabs(raw_target):
            raise SmokeError(f"bundle contains an absolute symlink: {entry}")
        try:
            target = entry.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise SmokeError(
                f"bundle contains a broken or cyclic symlink: {entry}"
            ) from error
        if not _is_within(target, root):
            raise SmokeError(f"bundle contains an escaping symlink: {entry}")


def _runtime_library_family(libs: Path, stem: str, label: str) -> LibraryFamily:
    try:
        entries = sorted(libs.glob(f"{stem}*"))
    except OSError as error:
        raise SmokeError(f"could not inspect bundled {stem} family") from error
    if not entries:
        raise SmokeError(f"bundle is missing required {label} family {stem}*")

    real_files: set[Path] = set()
    versioned: list[tuple[tuple[int, ...], Path]] = []
    version_re = re.compile(rf"{re.escape(stem)}[.]([0-9]+(?:[.][0-9]+)*)\Z")
    for entry in entries:
        real = resolve_regular_file(entry, f"bundle {stem} family", nonempty=True)
        real_files.add(real)
        if match := version_re.fullmatch(entry.name):
            versioned.append(
                (tuple(int(part) for part in match.group(1).split(".")), entry)
            )
    if len(real_files) != 1:
        raise SmokeError(f"bundle {label} family {stem}* has multiple real files")
    if not versioned:
        raise SmokeError(f"bundle {label} family {stem}* has no versioned SONAME path")

    majors = {version[0] for version, _ in versioned}
    if len(majors) != 1:
        raise SmokeError(f"bundle {label} family {stem}* has mixed CUDA majors")
    version, soname_path = min(
        versioned, key=lambda item: (len(item[0]), item[0], item[1].name)
    )
    return LibraryFamily(
        soname_path=soname_path,
        real=next(iter(real_files)),
        version=version,
    )


def _runtime_rtcx_providers(libs: Path) -> RtcxProviders:
    cudart = _runtime_library_family(libs, CUDA_RUNTIME_STEM, "CUDA runtime")
    providers = {
        stem: _runtime_library_family(libs, stem, "RTCX provider")
        for stem in RTCX_PROVIDER_STEMS
    }
    cuda_major = cudart.version[0]
    for stem, family in providers.items():
        if family.version[0] != cuda_major:
            raise SmokeError(
                f"bundle RTCX provider family {stem}* does not match libcudart CUDA "
                f"major {cuda_major}"
            )
    return RtcxProviders(
        cuda_major=cuda_major,
        cudart=cudart,
        nvrtc=providers["libnvrtc.so"],
        builtins=providers["libnvrtc-builtins.so"],
        nvjitlink=providers["libnvJitLink.so"],
    )


def _runtime_cufile_family(libs: Path) -> LibraryFamily:
    """Require the cuFile SONAME KvikIO dlopens, without treating so.0 as CUDA major."""

    soname_link = libs / "libcufile.so.0"
    if not soname_link.is_symlink():
        raise SmokeError("bundle is missing required libcufile.so.0 SONAME symlink")
    target = os.readlink(soname_link)
    if os.path.isabs(target) or "/" in target:
        raise SmokeError("bundled libcufile.so.0 must be a relative basename symlink")
    real = resolve_regular_file(soname_link, "bundle libcufile.so.0", nonempty=True)
    if real.parent != libs:
        raise SmokeError("bundled libcufile.so.0 must resolve inside libs/")
    return LibraryFamily(soname_path=soname_link, real=real, version=(0,))


def validate_bundle(value: str | os.PathLike[str]) -> Bundle:
    input_path = Path(value).expanduser()
    try:
        root = input_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SmokeError(f"--bundle does not resolve: {input_path}") from error
    if not root.is_dir():
        raise SmokeError(f"--bundle must name a directory: {input_path}")

    validate_bundle_symlinks(root)
    jars = sorted(path for path in root.glob("*.jar") if path.is_file())
    if len(jars) != 1:
        raise SmokeError("bundle must contain exactly one canonical Gluten bundle JAR")
    jar = resolve_regular_file(jars[0], "bundle JAR", nonempty=True)

    libs = root / "libs"
    ucx = libs / "ucx"
    if libs.is_symlink() or not libs.is_dir():
        raise SmokeError("bundle must contain the directory libs/")
    if ucx.is_symlink() or not ucx.is_dir():
        raise SmokeError("bundle must contain the directory libs/ucx/")
    if not any(ucx.iterdir()):
        raise SmokeError("bundle directory libs/ucx/ must be nonempty")
    libgluten = resolve_regular_file(
        libs / "libgluten.so", "bundle libs/libgluten.so", nonempty=True
    )
    rtcx = _runtime_rtcx_providers(libs)
    _runtime_cufile_family(libs)
    try:
        jar_info = read_bundle_jar(jar)
        native_info = read_native_build_info(libs / NATIVE_BUILD_INFO_NAME)
        validate_native_jar_identity(native_info, jar_info)
    except ArtifactMetadataError as error:
        raise SmokeError(str(error)) from error

    allowed_top_level = {jars[0].name, "libs"}
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in allowed_top_level
    )
    if unexpected:
        raise SmokeError(
            "bundle contains unexpected top-level entries: " + ", ".join(unexpected)
        )
    return Bundle(
        root=root,
        jar=jars[0],
        jar_info=jar_info,
        libgluten=libgluten,
        ucx=ucx,
        rtcx=rtcx,
    )


def qualification_target(info: BundleJarInfo) -> QualificationTarget:
    key = (info.spark_version, info.scala_binary, info.java_version)
    runtime = QUALIFICATION_TARGETS.get(key)
    if runtime is None:
        raise SmokeError(
            "bundle artifact is not in the clean-runtime qualification matrix: "
            f"Spark {info.spark_version}, Scala {info.scala_binary}, "
            f"JDK {info.java_version}"
        )
    return runtime


def clean_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    if extra:
        environment.update(extra)
    return environment


def _run_command(
    command: Sequence[str],
    *,
    capture_output: bool = False,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=capture_output,
        text=True,
        env=dict(environment) if environment is not None else None,
    )


def require_command(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise SmokeError(f"required command is missing: {name}")
    return resolved


def build_validation_image(runtime: QualificationTarget) -> str:
    result = _run_command(
        [
            "docker",
            "build",
            "--quiet",
            docker_platform_flag(),
            "--build-arg",
            f"SPARK_RUNTIME_IMAGE={runtime.image}",
            "--file",
            str(DOCKERFILE),
            str(SCRIPT_DIR),
        ],
        capture_output=True,
    )
    candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not candidates:
        raise SmokeError("Docker did not report the validation image")
    inspect = _run_command(
        ["docker", "image", "inspect", "--format", "{{.Id}}", candidates[-1]],
        capture_output=True,
    )
    image_id = inspect.stdout.strip()
    if not IMAGE_ID_RE.fullmatch(image_id):
        raise SmokeError(
            f"validation image did not resolve to a full image ID: {image_id}"
        )
    return image_id


def build_docker_run_command(
    bundle: Bundle, caller_properties: Path | None, image_id: str
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        docker_platform_flag(),
        "--user",
        "0:0",
        "--gpus",
        "1",
        "--mount",
        f"type=bind,src={bundle.root},dst={BUNDLE_MOUNT},readonly",
    ]
    if caller_properties is not None:
        command.extend(
            [
                "--mount",
                "type=bind,"
                f"src={caller_properties},dst={CALLER_PROPERTIES_MOUNT},readonly",
            ]
        )
    command.append(image_id)
    return command


def run_outer(bundle_value: str, spark_conf_value: str | None) -> None:
    # All caller input is validated before the first Docker command.
    bundle = validate_bundle(bundle_value)
    runtime = qualification_target(bundle.jar_info)
    caller_properties = None
    if spark_conf_value is not None:
        caller_properties = resolve_regular_file(
            spark_conf_value, "--spark_conf_file", nonempty=False
        )
    require_command("docker")

    if caller_properties is not None:
        print("Runtime smoke custom Spark properties: supplied read-only")
    print(
        f"Runtime smoke qualification target: Spark {runtime.spark_version}; "
        f"Scala {runtime.scala_binary}; JDK {runtime.java_version}; "
        "local[1]; shuffle partitions=1; AQE=false; ANSI=false; UI=false; "
        "native GPU required"
    )
    image_id = build_validation_image(runtime)
    _run_command(build_docker_run_command(bundle, caller_properties, image_id))


def _external_provider(name: str, resolved: Path | None, bundle_root: Path) -> bool:
    if name == "linux-vdso.so.1":
        return resolved is None
    if resolved is None or _is_within(resolved, bundle_root):
        return False

    glibc_names = {
        "libanl.so.1",
        "libBrokenLocale.so.1",
        "libc.so.6",
        "libdl.so.2",
        "libm.so.6",
        "libnsl.so.1",
        "libpthread.so.0",
        "libresolv.so.2",
        "librt.so.1",
        "libthread_db.so.1",
        "libutil.so.1",
    }
    system_roots = tuple(
        path.resolve()
        for path in (
            Path("/lib"),
            Path("/lib64"),
            Path("/usr/lib"),
            Path("/usr/lib64"),
        )
        if path.exists()
    )
    if (
        name in glibc_names
        or name in GLIBC_LOADERS
        or re.fullmatch(r"libnss_[A-Za-z0-9_-]+[.]so[.]2", name)
        or re.fullmatch(r"ld-linux[^/]*[.]so[.][12]", name)
    ):
        return any(_is_within(resolved, root) for root in system_roots)

    java_names = {
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
    java_home_value = os.environ.get("JAVA_HOME")
    if name in java_names and java_home_value:
        java_home = Path(java_home_value).resolve()
        return _is_within(resolved, java_home)

    if (
        re.fullmatch(r"libcuda[.]so(?:[.][0-9]+)*", name)
        or re.fullmatch(r"libnvidia-[A-Za-z0-9_-]+[.]so(?:[.][0-9]+)*", name)
        or re.fullmatch(r"libnvcuvid[.]so(?:[.][0-9]+)*", name)
    ):
        driver_roots = list(system_roots)
        for path in (Path("/usr/local/nvidia"), Path("/usr/local/cuda/compat")):
            if path.exists():
                driver_roots.append(path.resolve())
        return any(_is_within(resolved, root) for root in driver_roots)
    return False


def _parse_ldd_line(line: str) -> tuple[str, Path | None] | None:
    stripped = line.strip()
    if not stripped or stripped == "statically linked":
        return None
    if "=>" in stripped:
        name, value = (part.strip() for part in stripped.split("=>", 1))
        if value.startswith("not found"):
            raise SmokeError(f"unresolved runtime dependency: {name}")
        path_value = value.split(" ", 1)[0]
        if not path_value.startswith("/"):
            raise SmokeError(f"could not parse ldd resolution: {stripped}")
        return name, Path(path_value).resolve(strict=True)
    first = stripped.split(" ", 1)[0]
    if first == "linux-vdso.so.1":
        return first, None
    if first.startswith("/"):
        path = Path(first).resolve(strict=True)
        return path.name, path
    raise SmokeError(f"could not parse ldd output: {stripped}")


def validate_elf_closure(bundle: Bundle) -> None:
    environment = clean_environment()
    real_elfs: list[Path] = []
    for path in _walk_entries(bundle.root):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    real_elfs.append(path)
        except OSError as error:
            raise SmokeError(f"could not read bundle file: {path}") from error
    if not real_elfs:
        raise SmokeError("bundle contains no real ELF files")

    for elf in real_elfs:
        try:
            result = _run_command(
                ["ldd", str(elf)], capture_output=True, environment=environment
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "ldd failed").strip()
            raise SmokeError(f"ldd failed for {elf}: {detail}") from error
        for line in result.stdout.splitlines():
            parsed = _parse_ldd_line(line)
            if parsed is None:
                continue
            name, resolved = parsed
            if resolved is not None and _is_within(resolved, bundle.root):
                continue
            if not _external_provider(name, resolved, bundle.root):
                shown = str(resolved) if resolved is not None else "unresolved"
                raise SmokeError(
                    f"runtime dependency escaped the bundle/provider boundary: "
                    f"{elf.name}: {name} -> {shown}"
                )


def _mapped_library_paths(stem: str) -> set[Path]:
    mapped: set[Path] = set()
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SmokeError("could not inspect clean-runtime library mappings") from error
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        value = fields[5].removesuffix(" (deleted)")
        path = Path(value)
        if path.name.startswith(stem):
            mapped.add(path.resolve(strict=False))
    return mapped


def validate_rtcx_providers(bundle: Bundle) -> None:
    """Compile one kernel with bundled NVRTC and load bundled nvJitLink.

    This is a provider-usability check, not a CUDA kernel launch. The subsequent
    Spark smoke proves one GPU execution path; application-specific RTCX call
    sites and final-image driver compatibility remain consumer qualification.
    """

    providers = bundle.rtcx
    try:
        nvrtc = ctypes.CDLL(str(providers.nvrtc.soname_path), mode=ctypes.RTLD_LOCAL)
        nvrtc_error_string = nvrtc.nvrtcGetErrorString
        nvrtc_error_string.argtypes = [ctypes.c_int]
        nvrtc_error_string.restype = ctypes.c_char_p

        def check_nvrtc(result: int, operation: str) -> None:
            if result == 0:
                return
            value = nvrtc_error_string(result)
            detail = value.decode("utf-8", errors="replace") if value else str(result)
            raise SmokeError(f"bundled NVRTC {operation} failed: {detail}")

        nvrtc_version = nvrtc.nvrtcVersion
        nvrtc_version.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        nvrtc_version.restype = ctypes.c_int
        nvrtc_major = ctypes.c_int()
        nvrtc_minor = ctypes.c_int()
        check_nvrtc(
            nvrtc_version(ctypes.byref(nvrtc_major), ctypes.byref(nvrtc_minor)),
            "version query",
        )

        create_program = nvrtc.nvrtcCreateProgram
        create_program.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        create_program.restype = ctypes.c_int
        compile_program = nvrtc.nvrtcCompileProgram
        compile_program.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        compile_program.restype = ctypes.c_int
        get_ptx_size = nvrtc.nvrtcGetPTXSize
        get_ptx_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        get_ptx_size.restype = ctypes.c_int
        get_ptx = nvrtc.nvrtcGetPTX
        get_ptx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        get_ptx.restype = ctypes.c_int
        destroy_program = nvrtc.nvrtcDestroyProgram
        destroy_program.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        destroy_program.restype = ctypes.c_int

        program = ctypes.c_void_p()
        check_nvrtc(
            create_program(
                ctypes.byref(program),
                NVRTC_PROBE_SOURCE,
                b"rtcx-provider-smoke.cu",
                0,
                None,
                None,
            ),
            "program creation",
        )
        completed = False
        try:
            check_nvrtc(compile_program(program, 0, None), "compilation")
            ptx_size = ctypes.c_size_t()
            check_nvrtc(get_ptx_size(program, ctypes.byref(ptx_size)), "PTX sizing")
            if ptx_size.value <= 1:
                raise SmokeError("bundled NVRTC produced empty PTX")
            ptx = ctypes.create_string_buffer(ptx_size.value)
            check_nvrtc(
                get_ptx(program, ctypes.cast(ptx, ctypes.c_void_p)),
                "PTX retrieval",
            )

            mapped_builtins = _mapped_library_paths("libnvrtc-builtins.so")
            expected_builtins = providers.builtins.real.resolve(strict=True)
            if mapped_builtins != {expected_builtins}:
                shown = ", ".join(str(path) for path in sorted(mapped_builtins))
                raise SmokeError(
                    "NVRTC did not map exactly the bundled builtins provider: "
                    + (shown or "none")
                )
            if not _is_within(expected_builtins, bundle.root):
                raise SmokeError("NVRTC builtins provider escaped the bundle")
            completed = True
        finally:
            destroy_result = destroy_program(ctypes.byref(program))
            if completed:
                check_nvrtc(destroy_result, "program destruction")

        nvjitlink = ctypes.CDLL(
            str(providers.nvjitlink.soname_path), mode=ctypes.RTLD_LOCAL
        )
        nvjitlink_version = nvjitlink.nvJitLinkVersion
        nvjitlink_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        nvjitlink_version.restype = ctypes.c_int
        nvjitlink_major = ctypes.c_uint()
        nvjitlink_minor = ctypes.c_uint()
        result = nvjitlink_version(
            ctypes.byref(nvjitlink_major), ctypes.byref(nvjitlink_minor)
        )
        if result != 0:
            raise SmokeError(f"bundled nvJitLink version query failed: {result}")

        nvrtc_pair = (nvrtc_major.value, nvrtc_minor.value)
        nvjitlink_pair = (nvjitlink_major.value, nvjitlink_minor.value)
        if nvrtc_pair != nvjitlink_pair:
            raise SmokeError(
                f"bundled NVRTC {nvrtc_pair[0]}.{nvrtc_pair[1]} and nvJitLink "
                f"{nvjitlink_pair[0]}.{nvjitlink_pair[1]} do not match"
            )
        if nvrtc_pair[0] != providers.cuda_major:
            raise SmokeError(
                f"bundled NVRTC reports CUDA major {nvrtc_pair[0]}, expected "
                f"{providers.cuda_major}"
            )
    except SmokeError:
        raise
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise SmokeError(f"could not use bundled RTCX providers: {error}") from error

    print(
        f"Runtime smoke RTCX providers: CUDA {nvrtc_pair[0]}.{nvrtc_pair[1]}; "
        f"NVRTC PTX bytes={ptx_size.value}"
    )


def smoke_owned_spark_conf(jar: Path) -> tuple[tuple[str, str], ...]:
    libs = BUNDLE_MOUNT / "libs"
    libgluten = libs / "libgluten.so"
    return (
        ("spark.sql.shuffle.partitions", "1"),
        ("spark.sql.adaptive.enabled", "false"),
        ("spark.sql.ansi.enabled", "false"),
        ("spark.ui.enabled", "false"),
        ("spark.plugins", PLUGIN_CLASS),
        ("spark.shuffle.manager", SHUFFLE_MANAGER_CLASS),
        ("spark.memory.offHeap.enabled", "true"),
        ("spark.memory.offHeap.size", "1g"),
        ("spark.gluten.loadLibFromJar", "false"),
        ("spark.gluten.sql.columnar.libpath", str(libgluten)),
        ("spark.gluten.sql.columnar.executor.libpath", str(libgluten)),
        ("spark.driver.extraClassPath", str(jar)),
        ("spark.executor.extraClassPath", str(jar)),
        ("spark.driver.extraLibraryPath", str(libs)),
        ("spark.executor.extraLibraryPath", str(libs)),
        ("spark.executorEnv.UCX_MODULE_DIR", str(libs / "ucx")),
        ("spark.gluten.sql.columnar.cudf", "true"),
        (
            "spark.gluten.sql.columnar.backend.velox.cudf.enableValidation",
            "false",
        ),
        (
            "spark.gluten.sql.columnar.backend.velox.cudf.allow_cpu_fallback",
            "true",
        ),
        ("spark.gluten.sql.columnar.backend.velox.glogSeverityLevel", "0"),
        ("spark.gluten.sql.debug.cudf", "true"),
    )


def build_spark_submit_command(jar: Path, caller_properties: Path | None) -> list[str]:
    command = ["spark-submit"]
    if caller_properties is not None:
        command.extend(["--properties-file", str(caller_properties)])
    command.extend(["--master", "local[1]", "--jars", str(jar)])
    for key, value in smoke_owned_spark_conf(jar):
        command.extend(["--conf", f"{key}={value}"])
    command.append(str(QUERY_SCRIPT))
    return command


def adapted_cudf_operator(output: str) -> str | None:
    after_adaptation = False
    for line in output.splitlines():
        if "Operators before adapting for cuDF" in line:
            after_adaptation = False
        if "Operators after adapting for cuDF" in line:
            after_adaptation = True
            continue
        if after_adaptation and (marker := ADAPTED_CUDF_OPERATOR_RE.search(line)):
            return marker.group(1)
    return None


def run_container() -> None:
    bundle = validate_bundle(BUNDLE_MOUNT)
    runtime = qualification_target(bundle.jar_info)
    caller_properties = (
        CALLER_PROPERTIES_MOUNT if CALLER_PROPERTIES_MOUNT.exists() else None
    )
    if caller_properties is not None:
        caller_properties = resolve_regular_file(
            caller_properties, "mounted Spark properties file", nonempty=False
        )

    require_command("jar")
    require_command("ldd")
    _run_command(["jar", "tf", str(bundle.jar)], capture_output=True)
    validate_elf_closure(bundle)
    validate_rtcx_providers(bundle)

    if caller_properties is not None:
        print("Runtime smoke custom Spark properties: supplied read-only")
    print(
        f"Runtime smoke qualification target: Spark {runtime.spark_version}; "
        f"Scala {runtime.scala_binary}; JDK {runtime.java_version}; "
        "effective smoke-owned settings: master=local[1], "
        "shuffle partitions=1, AQE=false, ANSI=false, UI=false, "
        "Gluten/cuDF=true, cuDF validation=false, CPU fallback=true, "
        "native INFO logging=true"
    )
    environment = clean_environment(
        {"UCX_MODULE_DIR": str(BUNDLE_MOUNT / "libs" / "ucx")}
    )
    try:
        result = _run_command(
            build_spark_submit_command(bundle.jar, caller_properties),
            capture_output=True,
            environment=environment,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)
        raise
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    marker = adapted_cudf_operator(result.stdout + "\n" + result.stderr)
    if marker is None:
        raise SmokeError("cuDF post-adaptation output has no native reduction")
    print(f"Runtime smoke adapted cuDF operator: {marker}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean Spark-Gluten Velox GPU runtime-bundle smoke."
    )
    parser.add_argument("--bundle", required=True, help="Path to deploy/ bundle")
    parser.add_argument(
        "--spark_conf_file",
        help="Trusted caller Spark properties file (mounted read-only)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        # Container-only entrypoint; callers never pass this token.
        if arguments == ["--container-runtime"]:
            run_container()
        else:
            parsed = parse_args(arguments)
            run_outer(parsed.bundle, parsed.spark_conf_file)
        return 0
    except SmokeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"ERROR: command failed with exit code {error.returncode}: "
            + " ".join(str(part) for part in error.cmd),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
