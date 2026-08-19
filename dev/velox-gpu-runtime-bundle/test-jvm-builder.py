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

"""Hermetic tests for the Maven-only, one-variant JVM bundle entrypoint."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "build-velox-jvm-bundle.py"
SPEC = importlib.util.spec_from_file_location("build_velox_jvm_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)

GLUTEN_REVISION = "a" * 40
VELOX_REVISION = "b" * 40


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_pom(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <properties>
    <scala.binary.version>2.12</scala.binary.version>
  </properties>
  <profiles>
    <profile>
      <id>scala-2.12</id>
      <properties>
        <scala.version>2.12.15</scala.version>
        <scala.binary.version>2.12</scala.binary.version>
      </properties>
    </profile>
    <profile>
      <id>scala-2.13</id>
      <properties>
        <scala.version>2.13.17</scala.version>
        <scala.binary.version>2.13</scala.binary.version>
      </properties>
    </profile>
    <profile>
      <id>spark-3.5</id>
      <properties>
        <sparkbundle.version>3.5</sparkbundle.version>
        <sparkshim.artifactId>spark-sql-columnar-shims-spark35</sparkshim.artifactId>
        <spark.version>3.5.5</spark.version>
      </properties>
    </profile>
    <profile>
      <id>spark-4.0</id>
      <properties>
        <sparkbundle.version>4.0</sparkbundle.version>
        <sparkshim.artifactId>spark-sql-columnar-shims-spark40</sparkshim.artifactId>
        <spark.version>4.0.0</spark.version>
        <scala.binary.version>2.13</scala.binary.version>
      </properties>
    </profile>
  </profiles>
</project>
""",
        encoding="utf-8",
    )


def _write_bundle_jar(
    target: Path,
    *,
    suffix: str = "linux_amd64-1.6.0-SNAPSHOT",
    spark_line: str = "3.5",
    spark_version: str = "3.5.5",
    scala_binary: str = "2.12",
    scala_version: str = "2.12.15",
    gluten_revision: str = GLUTEN_REVISION,
    velox_revision: str = VELOX_REVISION,
) -> Path:
    jar = target / (
        f"gluten-velox-bundle-spark{spark_line}_{scala_binary}-{suffix}.jar"
    )
    shim = spark_line.replace(".", "")
    provider = f"org.apache.gluten.sql.shims.spark{shim}.SparkShimProvider"
    properties = (
        "backend_type=velox\n"
        "java_version=17\n"
        f"scala_version={scala_version}\n"
        f"spark_version={spark_version}\n"
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


class FakeRunner:
    def __init__(self, root: Path, velox: Path):
        self.root = root.resolve()
        self.velox = velox.resolve()
        self.commands: list[tuple[list[str], Path, dict[str, str] | None]] = []
        self.streamed_commands: list[list[str]] = []
        self.dirty: set[Path] = set()
        self.java_version = "17"
        self.maven_callback = self._default_maven_callback

    @staticmethod
    def _result(command: list[str], stdout: str = "", stderr: str = ""):
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    def _default_maven_callback(self, command: list[str], environment: dict[str, str]):
        version = next(
            item.split("=", 1)[1]
            for item in command
            if item.startswith("-Dspark.version=")
        )
        scala_binary = "2.13" if "scala-2.13" in " ".join(command) else "2.12"
        scala_version = "2.13.17" if scala_binary == "2.13" else "2.12.15"
        _write_bundle_jar(
            self.root / "package" / "target",
            spark_version=version,
            scala_binary=scala_binary,
            scala_version=scala_version,
            gluten_revision=environment["GLUTEN_BUILD_INFO_REVISION"],
            velox_revision=environment["GLUTEN_BUILD_INFO_VELOX_REVISION"],
        )

    def __call__(
        self,
        command,
        *,
        cwd: Path,
        environment: dict[str, str] | None,
        capture_output: bool = True,
    ):
        command = [str(item) for item in command]
        cwd = Path(cwd).resolve()
        captured_environment = None if environment is None else dict(environment)
        self.commands.append((command, cwd, captured_environment))
        if not capture_output:
            self.streamed_commands.append(command)
        executable = Path(command[0]).name
        if executable == "java":
            return self._result(
                command,
                stderr=f"    java.specification.version = {self.java_version}\n",
            )
        if executable == "javac":
            return self._result(command, stdout=f"javac {self.java_version}.0.1\n")
        if command[0] == "git":
            source = Path(command[command.index("-C") + 1]).resolve()
            if "--is-inside-work-tree" in command:
                return self._result(command, stdout="true\n")
            if "--verify" in command:
                revision = GLUTEN_REVISION if source == self.root else VELOX_REVISION
                return self._result(command, stdout=revision + "\n")
            if "status" in command:
                output = " M fixture\n" if source in self.dirty else ""
                return self._result(command, stdout=output)
        if Path(command[0]).resolve() == (self.root / "build" / "mvn").resolve():
            assert captured_environment is not None
            self.maven_callback(command, captured_environment)
            return self._result(command, stdout="BUILD SUCCESS\n")
        raise AssertionError(f"unexpected command: {command}")


class JvmBundleBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "gluten"
        self.velox = self.base / "velox"
        self.output = self.base / "output"
        self.java_home = self.base / "jdk17"
        (self.root / "package" / "target").mkdir(parents=True)
        self.velox.mkdir()
        self.output.mkdir()
        _write_pom(self.root / "pom.xml")
        _write_executable(self.root / "build" / "mvn")
        _write_executable(self.java_home / "bin" / "java")
        _write_executable(self.java_home / "bin" / "javac")
        self.runner = FakeRunner(self.root, self.velox)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **overrides):
        arguments = {
            "spark_profile": "spark-3.5",
            "scala_profile": "scala-2.12",
            "velox_home": self.velox,
            "output_dir": self.output,
            "gluten_root": self.root,
            "java_home": self.java_home,
            "runner": self.runner,
        }
        arguments.update(overrides)
        return BUILDER.build_jvm_bundle(**arguments)

    def maven_commands(self):
        return [item for item in self.runner.commands if Path(item[0][0]).name == "mvn"]

    def test_default_build_runs_only_one_variant_maven_reactor(self):
        result = self.build()
        self.assertEqual(result.spark_version, "3.5.5")
        self.assertEqual(result.scala_binary, "2.12")
        self.assertEqual(
            [path.name for path in self.output.iterdir()], [result.jar.name]
        )
        commands = self.maven_commands()
        self.assertEqual(len(commands), 1)
        command, _, environment = commands[0]
        self.assertEqual(self.runner.streamed_commands, [command])
        self.assertEqual(command[1:6], ["-pl", "package", "-am", "clean", "package"])
        self.assertIn("-Pbackends-velox,spark-3.5,scala-2.12,java-17", command)
        self.assertIn("-DskipTests", command)
        self.assertNotIn("-Dmaven.test.skip=true", command)
        self.assertNotIn("ALL", " ".join(command))
        self.assertNotRegex(" ".join(command), r"buildbundle|cmake|make")
        assert environment is not None
        self.assertEqual(environment["VELOX_HOME"], str(self.velox.resolve()))
        self.assertEqual(environment["GLUTEN_BUILD_INFO_REVISION"], GLUTEN_REVISION)
        self.assertEqual(
            environment["GLUTEN_BUILD_INFO_VELOX_REVISION"], VELOX_REVISION
        )

    def test_exact_vendor_coordinate_within_shim_is_forwarded(self):
        result = self.build(spark_version="3.5.5-vendor-1")
        self.assertEqual(result.spark_version, "3.5.5-vendor-1")
        command = self.maven_commands()[0][0]
        self.assertIn("-Dspark.version=3.5.5-vendor-1", command)

    def test_crossed_exact_coordinate_fails_before_maven(self):
        with self.assertRaisesRegex(
            BUILDER.JvmBundleBuildError, "outside selected shim"
        ):
            self.build(spark_version="4.0.2-vendor-1")
        self.assertEqual(self.maven_commands(), [])

    def test_unsupported_shim_fails_before_maven(self):
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "unsupported Spark"):
            self.build(spark_profile="spark-3.6")
        self.assertEqual(self.maven_commands(), [])

    def test_current_source_profiles_include_spark41_scala213_pair(self):
        selected = BUILDER.select_source_profiles(
            BUILDER.GLUTEN_ROOT / "pom.xml", "spark-4.1", "scala-2.13"
        )
        self.assertEqual("4.1", selected.spark_line)
        self.assertEqual("2.13", selected.scala_binary)

    def test_spark35_accepts_declared_scala213_profile(self):
        result = self.build(scala_profile="scala-2.13")
        self.assertEqual(result.spark_version, "3.5.5")
        self.assertEqual(result.scala_binary, "2.13")
        command = self.maven_commands()[0][0]
        self.assertIn("-Pbackends-velox,spark-3.5,scala-2.13,java-17", command)

    def test_incompatible_scala_profile_fails_before_maven(self):
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "supports Scala 2.13"):
            self.build(spark_profile="spark-4.0", scala_profile="scala-2.12")
        self.assertEqual(self.maven_commands(), [])

    def test_dirty_source_fails_before_maven(self):
        self.runner.dirty.add(self.velox.resolve())
        with self.assertRaisesRegex(
            BUILDER.JvmBundleBuildError, "Velox source must be clean"
        ):
            self.build()
        self.assertEqual(self.maven_commands(), [])

    def test_git_probes_trust_only_each_resolved_source(self):
        self.build()
        git_commands = [
            command for command, _, _ in self.runner.commands if command[0] == "git"
        ]
        self.assertEqual(len(git_commands), 6)
        for command in git_commands:
            source = Path(command[command.index("-C") + 1]).resolve()
            self.assertIn(source, (self.root.resolve(), self.velox.resolve()))
            self.assertEqual(
                command[:5],
                ["git", "-c", f"safe.directory={source}", "-C", str(source)],
            )
            self.assertNotIn("--global", command)

    def test_real_git_foreign_owner_emulation_preserves_clean_check(self):
        source = self.base / "foreign-owned-source"
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "config", "user.name", "fixture"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "config", "user.email", "fixture@example.com"],
            check=True,
        )
        tracked = source / "tracked"
        tracked.write_text("clean\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "tracked"], check=True)
        subprocess.run(
            ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        expected = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

        with mock.patch.dict(
            os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1"}, clear=False
        ):
            self.assertEqual(
                expected,
                BUILDER._clean_git_head(source, "Fixture", BUILDER._run_command),
            )
            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(
                BUILDER.JvmBundleBuildError, "Fixture source must be clean"
            ):
                BUILDER._clean_git_head(source, "Fixture", BUILDER._run_command)

    def test_non_jdk17_fails_before_git_or_maven(self):
        self.runner.java_version = "21"
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "requires JDK 17"):
            self.build()
        self.assertEqual(self.maven_commands(), [])

    def test_symlinked_java_home_resolves_to_jdk17(self):
        java_home_link = self.base / "java-17-openjdk"
        java_home_link.symlink_to(self.java_home, target_is_directory=True)
        self.build(java_home=java_home_link)
        _, _, environment = self.maven_commands()[0]
        assert environment is not None
        self.assertEqual(environment["JAVA_HOME"], str(self.java_home.resolve()))

    def test_multiple_matching_jars_are_rejected(self):
        def write_two(command, environment):
            _write_bundle_jar(self.root / "package" / "target", suffix="one")
            _write_bundle_jar(self.root / "package" / "target", suffix="two")

        self.runner.maven_callback = write_two
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "found 2"):
            self.build()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_crossed_jar_metadata_is_rejected(self):
        def write_crossed(command, environment):
            _write_bundle_jar(
                self.root / "package" / "target",
                spark_line="3.5",
                spark_version="4.0.2",
            )

        self.runner.maven_callback = write_crossed
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "found 0"):
            self.build()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_nonempty_output_fails_before_any_command(self):
        (self.output / "stale").write_text("fixture", encoding="utf-8")
        with self.assertRaisesRegex(BUILDER.JvmBundleBuildError, "must be empty"):
            self.build()
        self.assertEqual(self.runner.commands, [])

    def test_cli_import_does_not_create_source_bytecode(self):
        cli_root = self.base / "cli-source"
        helper_dir = cli_root / "dev" / "velox-gpu-runtime-bundle"
        helper_dir.mkdir(parents=True)
        shutil.copy2(SCRIPT, cli_root / "dev" / SCRIPT.name)
        shutil.copy2(
            SCRIPT.parent / "velox-gpu-runtime-bundle" / "artifact_metadata.py",
            helper_dir / "artifact_metadata.py",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        subprocess.run(
            [sys.executable, str(cli_root / "dev" / SCRIPT.name), "--help"],
            cwd=cli_root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual([], list(cli_root.rglob("__pycache__")))

    def test_build_info_accepts_verified_archive_revisions_without_git(self):
        archive_root = self.base / "archive-source"
        script = archive_root / "dev" / "gluten-build-info.sh"
        script.parent.mkdir(parents=True)
        (archive_root / "gluten-core").mkdir()
        shutil.copy2(SCRIPT.parent / "gluten-build-info.sh", script)
        environment = os.environ.copy()
        environment.update(
            {
                "GLUTEN_BUILD_INFO_REVISION": GLUTEN_REVISION,
                "GLUTEN_BUILD_INFO_VELOX_REVISION": VELOX_REVISION,
                "VELOX_HOME": str(archive_root / "velox-without-git"),
            }
        )

        subprocess.run(
            [
                "bash",
                str(script),
                "--version",
                "fixture",
                "--backend",
                "velox",
                "--java",
                "17",
                "--scala",
                "2.12.15",
                "--spark",
                "3.5.5",
                "--revision",
                "true",
            ],
            cwd=archive_root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        build_info = (
            archive_root
            / "gluten-core"
            / "target"
            / "generated-resources"
            / "gluten-build-info.properties"
        ).read_text(encoding="utf-8")
        self.assertIn(f"revision={GLUTEN_REVISION}\n", build_info)
        self.assertIn(f"velox_revision={VELOX_REVISION}\n", build_info)

        environment["GLUTEN_BUILD_INFO_REVISION"] = "short"
        failed = subprocess.run(
            ["bash", str(script), "--revision", "true"],
            cwd=archive_root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(0, failed.returncode)
        self.assertIn("full lowercase Git SHA", failed.stderr)


if __name__ == "__main__":
    unittest.main()
