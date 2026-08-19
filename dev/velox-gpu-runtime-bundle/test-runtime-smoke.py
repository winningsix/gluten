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

"""Hermetic bundle, RTCX-provider, and command tests for the runtime smoke."""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime_smoke = load_module("runtime_smoke", "runtime-smoke.py")
runtime_query = load_module("runtime_query", "runtime-query.py")

GLUTEN_REVISION = "a" * 40
VELOX_REVISION = "b" * 40


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class FakeNvrtc:
    def __init__(self, *, version=(12, 4), compile_result=0):
        self.destroy_calls = 0
        self.nvrtcGetErrorString = FakeFunction(lambda result: b"fixture error")
        self.nvrtcVersion = FakeFunction(
            lambda major, minor: self._version(major, minor, version)
        )
        self.nvrtcCreateProgram = FakeFunction(self._create_program)
        self.nvrtcCompileProgram = FakeFunction(
            lambda program, count, options: compile_result
        )
        self.nvrtcGetPTXSize = FakeFunction(self._ptx_size)
        self.nvrtcGetPTX = FakeFunction(self._ptx)
        self.nvrtcDestroyProgram = FakeFunction(self._destroy_program)

    @staticmethod
    def _version(major, minor, version):
        major._obj.value, minor._obj.value = version
        return 0

    @staticmethod
    def _create_program(program, source, name, header_count, headers, names):
        program._obj.value = 1
        return 0

    @staticmethod
    def _ptx_size(program, size):
        size._obj.value = 8
        return 0

    @staticmethod
    def _ptx(program, destination):
        ctypes.memmove(destination, b"PTXDATA\0", 8)
        return 0

    def _destroy_program(self, program):
        self.destroy_calls += 1
        program._obj.value = None
        return 0


class FakeNvJitLink:
    def __init__(self, version=(12, 4)):
        self.nvJitLinkVersion = FakeFunction(
            lambda major, minor: FakeNvrtc._version(major, minor, version)
        )


class FakeDocker:
    IMAGE_ID = "sha256:" + "a" * 64

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, command, *, capture_output=False, environment=None):
        call = list(command)
        self.calls.append(call)
        stdout = ""
        if call[:2] == ["docker", "build"]:
            stdout = self.IMAGE_ID + "\n"
        elif call[:3] == ["docker", "image", "inspect"]:
            stdout = self.IMAGE_ID + "\n"
        return subprocess.CompletedProcess(call, 0, stdout, "")


class RuntimeSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "deploy"
        (self.bundle / "libs" / "ucx").mkdir(parents=True)
        self.jar = self._write_bundle_jar()
        self._write_native_info()
        (self.bundle / "libs" / "libgluten.so.1").write_bytes(b"library")
        (self.bundle / "libs" / "libgluten.so").symlink_to("libgluten.so.1")
        (self.bundle / "libs" / "ucx" / "libuct_cuda.so").write_bytes(b"module")
        self._rtcx_family("libcudart.so", "libcudart.so.12", "libcudart.so.12.4.99")
        self._rtcx_family("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.12.4.99")
        self._rtcx_family(
            "libnvrtc-builtins.so",
            "libnvrtc-builtins.so.12.4",
            "libnvrtc-builtins.so.12.4.99",
        )
        self._rtcx_family(
            "libnvJitLink.so", "libnvJitLink.so.12", "libnvJitLink.so.12.4.99"
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_native_info(
        self,
        *,
        gluten_revision: str = GLUTEN_REVISION,
        velox_revision: str = VELOX_REVISION,
    ) -> None:
        (self.bundle / "libs" / "gluten-native-build-info.properties").write_text(
            "backend_type=velox\n"
            f"gluten_revision={gluten_revision}\n"
            f"velox_revision={velox_revision}\n",
            encoding="utf-8",
        )

    def _write_bundle_jar(
        self,
        *,
        spark_line: str = "3.5",
        spark_version: str = "3.5.5",
        scala_binary: str = "2.12",
        scala_version: str = "2.12.15",
        java_version: str = "17",
        gluten_revision: str = GLUTEN_REVISION,
        velox_revision: str = VELOX_REVISION,
    ) -> Path:
        jar = self.bundle / (
            f"gluten-velox-bundle-spark{spark_line}_{scala_binary}-test.jar"
        )
        provider = (
            "org.apache.gluten.sql.shims.spark"
            f"{spark_line.replace('.', '')}.SparkShimProvider"
        )
        properties = (
            "backend_type=velox\n"
            f"spark_version={spark_version}\n"
            f"scala_version={scala_version}\n"
            f"java_version={java_version}\n"
            f"revision={gluten_revision}\n"
            f"velox_revision={velox_revision}\n"
        )
        with zipfile.ZipFile(jar, "w") as archive:
            archive.writestr("gluten-build-info.properties", properties)
            archive.writestr(
                "META-INF/services/org.apache.gluten.sql.shims.SparkShimProvider",
                provider + "\n",
            )
            archive.writestr(provider.replace(".", "/") + ".class", b"fixture")
        return jar

    def _replace_bundle_jar(self, **kwargs) -> Path:
        self.jar.unlink()
        self.jar = self._write_bundle_jar(**kwargs)
        return self.jar

    def _rtcx_family(self, stem: str, soname: str, real_name: str):
        libs = self.bundle / "libs"
        (libs / real_name).write_bytes(b"ELF fixture")
        (libs / soname).symlink_to(real_name)
        (libs / stem).symlink_to(soname)

    def _run_outer(self, config=None):
        fake = FakeDocker()
        with mock.patch.object(
            runtime_smoke, "_run_command", side_effect=fake
        ), mock.patch.object(
            runtime_smoke, "require_command", return_value="/usr/bin/docker"
        ):
            runtime_smoke.run_outer(
                str(self.bundle), None if config is None else str(config)
            )
        return fake.calls

    def test_no_option_preserves_canonical_command(self):
        calls = self._run_outer()
        docker_build = calls[0]
        self.assertIn(
            "SPARK_RUNTIME_IMAGE=" "apache/spark:3.5.5-scala2.12-java17-python3-ubuntu",
            docker_build,
        )
        docker_run = calls[-1]
        self.assertEqual(docker_run[docker_run.index("--gpus") + 1], "1")
        self.assertNotIn(
            str(runtime_smoke.CALLER_PROPERTIES_MOUNT), " ".join(docker_run)
        )
        jar = runtime_smoke.BUNDLE_MOUNT / (
            "gluten-velox-bundle-spark3.5_2.12-test.jar"
        )
        spark_command = runtime_smoke.build_spark_submit_command(jar, None)
        self.assertNotIn("--properties-file", spark_command)
        self.assertEqual(spark_command[-1], str(runtime_smoke.QUERY_SCRIPT))

    def test_spark_4_variant_selects_its_exact_runtime_image(self):
        self._replace_bundle_jar(
            spark_line="4.0",
            spark_version="4.0.2",
            scala_binary="2.13",
            scala_version="2.13.16",
        )
        calls = self._run_outer()
        self.assertIn(
            "SPARK_RUNTIME_IMAGE=" "apache/spark:4.0.2-scala2.13-java17-python3-ubuntu",
            calls[0],
        )

    def test_unqualified_artifact_fails_before_docker(self):
        cases = (
            {"spark_version": "3.5.4"},
            {
                "spark_line": "4.0",
                "spark_version": "4.0.0",
                "scala_binary": "2.13",
                "scala_version": "2.13.16",
            },
        )
        for values in cases:
            with self.subTest(values=values):
                self._replace_bundle_jar(**values)
                with mock.patch.object(
                    runtime_smoke, "_run_command"
                ) as run_command, self.assertRaisesRegex(
                    runtime_smoke.SmokeError,
                    "not in the clean-runtime qualification matrix",
                ):
                    runtime_smoke.run_outer(str(self.bundle), None)
                run_command.assert_not_called()

    def test_bundle_rejects_filename_metadata_crossings_before_docker(self):
        cases = (
            {"spark_version": "4.0.0"},
            {"scala_version": "2.13.16"},
        )
        for values in cases:
            with self.subTest(values=values):
                self._replace_bundle_jar(**values)
                with mock.patch.object(
                    runtime_smoke, "_run_command"
                ) as run_command, self.assertRaises(runtime_smoke.SmokeError):
                    runtime_smoke.run_outer(str(self.bundle), None)
                run_command.assert_not_called()

    def test_bundle_rejects_native_jvm_revision_mismatch_before_docker(self):
        self._write_native_info(gluten_revision="c" * 40)
        with mock.patch.object(
            runtime_smoke, "_run_command"
        ) as run_command, self.assertRaisesRegex(
            runtime_smoke.SmokeError, "Gluten revisions do not match"
        ):
            runtime_smoke.run_outer(str(self.bundle), None)
        run_command.assert_not_called()

    def test_valid_config_is_read_only_and_precedes_smoke_owned_settings(self):
        config = self.root / "spark-defaults.conf"
        config.write_text(
            "spark.master local[99]\n"
            "spark.sql.shuffle.partitions 99\n"
            "spark.sql.ansi.enabled true\n"
            "spark.gluten.sql.columnar.backend.velox.cudf.enableValidation true\n"
            "spark.gluten.sql.columnar.backend.velox.cudf.allow_cpu_fallback false\n"
            "spark.gluten.sql.columnar.backend.velox.glogSeverityLevel 1\n"
            "spark.sql.session.timeZone UTC\n",
            encoding="utf-8",
        )
        calls = self._run_outer(config)
        docker_run = calls[-1]
        expected_mount = (
            f"type=bind,src={config.resolve()},"
            f"dst={runtime_smoke.CALLER_PROPERTIES_MOUNT},readonly"
        )
        self.assertIn(expected_mount, docker_run)

        jar = runtime_smoke.BUNDLE_MOUNT / (
            "gluten-velox-bundle-spark3.5_2.12-test.jar"
        )
        spark_command = runtime_smoke.build_spark_submit_command(
            jar, runtime_smoke.CALLER_PROPERTIES_MOUNT
        )
        properties_position = spark_command.index("--properties-file")
        master_position = spark_command.index("--master")
        smoke_owned_partition = "spark.sql.shuffle.partitions=1"
        self.assertLess(properties_position, master_position)
        self.assertGreater(
            spark_command.index(smoke_owned_partition), properties_position
        )
        self.assertIn("local[1]", spark_command)
        self.assertNotIn("local[99]", spark_command)
        self.assertNotIn("spark.sql.shuffle.partitions=99", spark_command)
        self.assertIn("spark.sql.ansi.enabled=false", spark_command)
        self.assertNotIn("spark.sql.ansi.enabled=true", spark_command)
        self.assertIn(
            "spark.gluten.sql.columnar.backend.velox.cudf.enableValidation=false",
            spark_command,
        )
        self.assertIn(
            "spark.gluten.sql.columnar.backend.velox.cudf.allow_cpu_fallback=true",
            spark_command,
        )
        self.assertIn(
            "spark.gluten.sql.columnar.backend.velox.glogSeverityLevel=0",
            spark_command,
        )
        self.assertEqual(spark_command[-1], str(runtime_smoke.QUERY_SCRIPT))

    def test_empty_readable_config_is_accepted(self):
        config = self.root / "empty-spark-defaults.conf"
        config.touch()
        calls = self._run_outer(config)
        expected_mount = (
            f"type=bind,src={config.resolve()},"
            f"dst={runtime_smoke.CALLER_PROPERTIES_MOUNT},readonly"
        )
        self.assertIn(expected_mount, calls[-1])

    def test_bad_config_fails_before_docker(self):
        missing = self.root / "missing.conf"
        directory = self.root / "config-dir"
        directory.mkdir()
        unreadable = self.root / "unreadable.conf"
        unreadable.write_text("", encoding="utf-8")
        unreadable.chmod(0)

        for config in (missing, directory, unreadable):
            with self.subTest(config=config), mock.patch.object(
                runtime_smoke, "_run_command"
            ) as run_command, self.assertRaises(runtime_smoke.SmokeError):
                runtime_smoke.run_outer(str(self.bundle), str(config))
            run_command.assert_not_called()

    def test_bundle_rejects_unsafe_symlink_before_docker(self):
        unsafe = self.bundle / "libs" / "unsafe.so"
        unsafe.symlink_to("/usr/lib/libunsafe.so")
        with mock.patch.object(
            runtime_smoke, "_run_command"
        ) as run_command, self.assertRaises(runtime_smoke.SmokeError):
            runtime_smoke.run_outer(str(self.bundle), None)
        run_command.assert_not_called()

    def test_bundle_rejects_missing_rtcx_family_before_docker(self):
        for path in (self.bundle / "libs").glob("libnvrtc-builtins.so*"):
            path.unlink()
        with mock.patch.object(
            runtime_smoke, "_run_command"
        ) as run_command, self.assertRaisesRegex(
            runtime_smoke.SmokeError, "libnvrtc-builtins"
        ):
            runtime_smoke.run_outer(str(self.bundle), None)
        run_command.assert_not_called()

    def test_bundle_rejects_rtcx_cuda_major_mismatch_before_docker(self):
        libs = self.bundle / "libs"
        for path in libs.glob("libnvJitLink.so*"):
            path.unlink()
        self._rtcx_family(
            "libnvJitLink.so", "libnvJitLink.so.11", "libnvJitLink.so.11.8.99"
        )
        with mock.patch.object(
            runtime_smoke, "_run_command"
        ) as run_command, self.assertRaisesRegex(
            runtime_smoke.SmokeError, "does not match libcudart CUDA major 12"
        ):
            runtime_smoke.run_outer(str(self.bundle), None)
        run_command.assert_not_called()

    def _probe_with_fakes(
        self,
        *,
        compile_result=0,
        nvrtc_version=(12, 4),
        nvjitlink_version=(12, 4),
        mapped_builtins=None,
    ):
        bundle = runtime_smoke.validate_bundle(self.bundle)
        fake_nvrtc = FakeNvrtc(version=nvrtc_version, compile_result=compile_result)
        fake_nvjitlink = FakeNvJitLink(version=nvjitlink_version)

        def load_library(path, mode=None):
            name = Path(path).name
            if name.startswith("libnvrtc.so"):
                return fake_nvrtc
            if name.startswith("libnvJitLink.so"):
                return fake_nvjitlink
            raise AssertionError(f"unexpected library load: {path}")

        mappings = (
            {bundle.rtcx.builtins.real} if mapped_builtins is None else mapped_builtins
        )
        with mock.patch.object(
            runtime_smoke.ctypes, "CDLL", side_effect=load_library
        ), mock.patch.object(
            runtime_smoke, "_mapped_library_paths", return_value=mappings
        ):
            runtime_smoke.validate_rtcx_providers(bundle)
        return fake_nvrtc

    def test_rtcx_probe_compiles_ptx_and_loads_matching_nvjitlink(self):
        fake_nvrtc = self._probe_with_fakes()
        self.assertEqual(1, fake_nvrtc.destroy_calls)

    def test_rtcx_probe_destroys_program_after_compile_failure(self):
        bundle = runtime_smoke.validate_bundle(self.bundle)
        fake_nvrtc = FakeNvrtc(compile_result=6)

        def load_library(path, mode=None):
            if Path(path).name.startswith("libnvrtc.so"):
                return fake_nvrtc
            return FakeNvJitLink()

        with mock.patch.object(
            runtime_smoke.ctypes, "CDLL", side_effect=load_library
        ), self.assertRaisesRegex(runtime_smoke.SmokeError, "compilation failed"):
            runtime_smoke.validate_rtcx_providers(bundle)
        self.assertEqual(1, fake_nvrtc.destroy_calls)

    def test_rtcx_probe_rejects_external_builtins_mapping(self):
        with self.assertRaisesRegex(
            runtime_smoke.SmokeError, "exactly the bundled builtins"
        ):
            self._probe_with_fakes(
                mapped_builtins={Path("/opt/cuda/libnvrtc-builtins.so.12.4.99")}
            )

    def test_rtcx_probe_rejects_nvjitlink_version_mismatch(self):
        with self.assertRaisesRegex(runtime_smoke.SmokeError, "do not match"):
            self._probe_with_fakes(nvjitlink_version=(12, 3))

    def test_query_and_native_acceptance_contract_is_fixed(self):
        self.assertEqual(
            runtime_query.QUERY,
            "SELECT sum(id) AS total FROM range(0, 32, 1, 1) WHERE id % 2 = 0",
        )
        self.assertEqual(runtime_query.EXPECTED_TOTAL, 240)
        self.assertIsNotNone(
            runtime_query.NATIVE_PLAN_RE.search("CudfFilterExecTransformer")
        )
        self.assertIsNone(runtime_query.NATIVE_PLAN_RE.search("FilterExec"))
        self.assertEqual(
            runtime_query.LIBGLUTEN,
            Path("/opt/gluten-deploy/libs/libgluten.so"),
        )
        required = dict(
            runtime_query._required_configuration(
                Path("/opt/gluten-deploy/gluten-bundle.jar")
            )
        )
        self.assertEqual(required["spark.sql.ansi.enabled"], "false")
        self.assertFalse(
            runtime_smoke._external_provider(
                "libcrypt.so.1", Path("/usr/lib/libcrypt.so.1"), self.bundle
            )
        )

    def test_query_runtime_identity_accepts_each_qualified_variant(self):
        cases = (
            ("3.5", "3.5.5", "2.12", "2.12.15"),
            ("4.0", "4.0.2", "2.13", "2.13.16"),
        )
        for spark_line, spark_version, scala_binary, scala_version in cases:
            with self.subTest(spark_version=spark_version):
                if self.jar.exists():
                    self.jar.unlink()
                self.jar = self._write_bundle_jar(
                    spark_line=spark_line,
                    spark_version=spark_version,
                    scala_binary=scala_binary,
                    scala_version=scala_version,
                )
                info = runtime_smoke.read_bundle_jar(self.jar)
                runtime_query._assert_runtime_identity(
                    info, spark_version, scala_version, "17"
                )

    def test_query_runtime_identity_rejects_spark_scala_and_jdk_mismatch(self):
        info = runtime_smoke.read_bundle_jar(self.jar)
        cases = (
            ("3.5.4", "2.12.15", "17", "expected Spark 3.5.5"),
            ("3.5.5", "2.13.16", "17", "expected Scala 2.12"),
            ("3.5.5", "2.12.15", "21", "expected JDK 17"),
        )
        for spark_version, scala_version, java_version, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                AssertionError, message
            ):
                runtime_query._assert_runtime_identity(
                    info, spark_version, scala_version, java_version
                )

    def test_native_operator_requires_post_adaptation_compute_operator(self):
        cpu_only = """\
Operators after adapting for cuDF: count [3]
  Operator: ID 0: CudfValueStream[0] 0
  Operator: ID 1: FilterProject
  Operator: ID 2: CudfToVelox
CudfFilterExecTransformer
"""
        self.assertIsNone(runtime_smoke.adapted_cudf_operator(cpu_only))

        gpu_compute = """\
Operators before adapting for cuDF: count [1]
  Operator: ID 0: Aggregation[2] 2
Operators after adapting for cuDF: count [3]
  Operator: ID 0: CudfFromVelox[2-from-velox] 0
  Operator: ID 1: CudfReduceSINGLE[2] 1
  Operator: ID 2: CudfToVelox[2-to-velox] 2
"""
        self.assertEqual(
            runtime_smoke.adapted_cudf_operator(gpu_compute), "CudfReduceSINGLE"
        )

    def test_validation_image_exposes_spark_submit_on_path(self):
        dockerfile = (HERE / "runtime-smoke.Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG SPARK_RUNTIME_IMAGE", dockerfile)
        self.assertIn("FROM ${SPARK_RUNTIME_IMAGE}", dockerfile)
        self.assertIn("COPY artifact_metadata.py runtime-smoke.py", dockerfile)
        self.assertIn('ENV PATH="${SPARK_HOME}/bin:${PATH}"', dockerfile)
        self.assertIn("command -v spark-submit", dockerfile)


if __name__ == "__main__":
    unittest.main()
