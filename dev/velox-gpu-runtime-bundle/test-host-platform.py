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

"""Hermetic tests for native host-platform mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "host_platform", HERE / "host_platform.py"
)
assert SPEC is not None and SPEC.loader is not None
host_platform = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = host_platform
SPEC.loader.exec_module(host_platform)


class HostPlatformTest(unittest.TestCase):
    def test_x86_64_aliases(self):
        for machine in ("x86_64", "amd64", "X86_64"):
            platform = host_platform.host_platform(machine=machine)
            self.assertEqual("linux/amd64", platform.docker_platform)
            self.assertEqual("amd64", platform.maven_arch)
            self.assertEqual(host_platform.EM_X86_64, platform.elf_machine)
            self.assertEqual("ld-linux-x86-64.so.2", platform.glibc_loader)
            self.assertIn(platform.glibc_loader, host_platform.GLIBC_LOADERS)

    def test_aarch64_aliases(self):
        for machine in ("aarch64", "arm64"):
            platform = host_platform.host_platform(machine=machine)
            self.assertEqual("linux/arm64", platform.docker_platform)
            self.assertEqual("aarch64", platform.maven_arch)
            self.assertEqual(host_platform.EM_AARCH64, platform.elf_machine)
            self.assertEqual("ld-linux-aarch64.so.1", platform.glibc_loader)

    def test_rejects_unsupported_machine(self):
        with self.assertRaisesRegex(host_platform.HostPlatformError, "ppc64le"):
            host_platform.host_platform(machine="ppc64le")

    def test_live_host_matches_uname(self):
        platform = host_platform.host_platform()
        self.assertIn(platform.docker_platform, {"linux/amd64", "linux/arm64"})
        self.assertEqual(
            host_platform.docker_platform_flag(),
            f"--platform={platform.docker_platform}",
        )
        self.assertTrue(host_platform.jar_os_suffix().startswith("linux_"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
