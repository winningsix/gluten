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
# See the LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

"""Map the native host machine to Docker, Maven, and ELF identities.

GPU packaging and smoke must run on the host architecture. This helper never
emits a ``--platform`` that differs from ``uname -m`` and does not encode CUDA
toolkit triples such as ``sbsa-linux``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys


class HostPlatformError(RuntimeError):
    """The host machine is unsupported or disagrees with a requested platform."""


# ELF e_machine values from the System V ABI.
EM_X86_64 = 62
EM_AARCH64 = 183

GLIBC_LOADERS = frozenset(
    {
        "ld-linux-aarch64.so.1",
        "ld-linux-x86-64.so.2",
        "ld64.so.1",
        "ld64.so.2",
    }
)


@dataclass(frozen=True)
class HostPlatform:
    machine: str
    docker_platform: str
    maven_arch: str
    elf_machine: int
    glibc_loader: str


def _normalize_machine(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "aarch64"
    raise HostPlatformError(
        f"unsupported host machine {machine!r}; native x86_64 or aarch64 is required"
    )


def host_platform(*, machine: str | None = None) -> HostPlatform:
    """Return the native host identity.

    ``machine`` is test-only. Production callers must omit it so the live
    ``uname -m`` (or ``os.uname``) is used.
    """

    raw = os.uname().machine if machine is None else machine
    normalized = _normalize_machine(raw)
    if normalized == "x86_64":
        return HostPlatform(
            machine="x86_64",
            docker_platform="linux/amd64",
            maven_arch="amd64",
            elf_machine=EM_X86_64,
            glibc_loader="ld-linux-x86-64.so.2",
        )
    return HostPlatform(
        machine="aarch64",
        docker_platform="linux/arm64",
        maven_arch="aarch64",
        elf_machine=EM_AARCH64,
        glibc_loader="ld-linux-aarch64.so.1",
    )


def jar_os_suffix(*, machine: str | None = None) -> str:
    """Maven ``os.full.name`` prefix used in the canonical bundle JAR."""

    platform = host_platform(machine=machine)
    return f"linux_{platform.maven_arch}"


def docker_platform_flag(*, machine: str | None = None) -> str:
    """``--platform=`` value that matches the host and refuses qemu."""

    return f"--platform={host_platform(machine=machine).docker_platform}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    platform = host_platform()
    if args in ([], ["--docker-platform"]):
        print(platform.docker_platform)
        return 0
    if args == ["--maven-arch"]:
        print(platform.maven_arch)
        return 0
    if args == ["--elf-machine"]:
        print(platform.elf_machine)
        return 0
    if args == ["--glibc-loader"]:
        print(platform.glibc_loader)
        return 0
    print(
        "usage: host_platform.py [--docker-platform|--maven-arch|"
        "--elf-machine|--glibc-loader]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
