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

"""Exercise the packager with a small, compiled ELF dependency graph.

The fixtures cover both content modes plus closure, RTCX provider families,
aliases, collisions, RUNPATH, and unsafe-symlink behavior without requiring a
Gluten build.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

PACKAGER_PATH = Path(__file__).parents[1] / "build-velox-gpu-runtime-bundle.py"
GLUTEN_REVISION = "1" * 40
VELOX_REVISION = "2" * 40
SPEC = importlib.util.spec_from_file_location(
    "velox_gpu_runtime_packager", PACKAGER_PATH
)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packager
SPEC.loader.exec_module(packager)


def run(command: list[str]):
    subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.sources = root / "sources"
        self.libdir = root / "lib"
        self.cuda = root / "cuda" / "targets" / "x86_64-linux" / "lib"
        self.ucx = root / "ucx"
        self.output = root / "deploy"
        for directory in (self.sources, self.libdir, self.cuda, self.ucx, self.output):
            directory.mkdir(parents=True)
        self.jar = root / "gluten-velox-bundle-spark3.5_2.12-1.6.0-SNAPSHOT.jar"
        self.write_jar(self.jar)
        self._build_default_libraries()

    @staticmethod
    def write_jar(
        path: Path,
        *,
        spark_version: str = "3.5.5",
        scala_version: str = "2.12.15",
        gluten_revision: str = GLUTEN_REVISION,
        velox_revision: str = VELOX_REVISION,
        backend_type: str = "velox",
    ) -> Path:
        spark_line = ".".join(spark_version.split(".")[:2])
        shim = spark_line.replace(".", "")
        provider = f"org.apache.gluten.sql.shims.spark{shim}.SparkShimProvider"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
            archive.writestr(
                "gluten-build-info.properties",
                f"backend_type={backend_type}\n"
                f"java_version=17\n"
                f"revision={gluten_revision}\n"
                f"scala_version={scala_version}\n"
                f"spark_version={spark_version}\n"
                f"velox_revision={velox_revision}\n",
            )
            archive.writestr(
                "META-INF/services/org.apache.gluten.sql.shims.SparkShimProvider",
                provider + "\n",
            )
            archive.writestr(provider.replace(".", "/") + ".class", b"class")
        return path

    def source(self, name: str, body: str) -> Path:
        path = self.sources / name
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def shared(
        self,
        output: Path,
        soname: str,
        source: Path,
        *,
        library_dirs: tuple[Path, ...] = (),
        libraries: tuple[str, ...] = (),
    ):
        command = [
            "gcc",
            "-shared",
            "-fPIC",
            f"-Wl,-soname,{soname}",
            "-o",
            str(output),
            str(source),
        ]
        effective_library_dirs = list(library_dirs)
        if output.name == "libgluten.so" and self.cuda not in effective_library_dirs:
            effective_library_dirs.append(self.cuda)
        for directory in effective_library_dirs:
            command.extend([f"-L{directory}", f"-Wl,-rpath,{directory}"])
        # Every fixture libgluten models the production CUDA runtime edge, even
        # when a focused test replaces its source to exercise another closure.
        if output.name == "libgluten.so":
            command.extend(["-Wl,--no-as-needed", "-lcudart", "-Wl,--as-needed"])
        command.extend(f"-l{library}" for library in libraries)
        run(command)

    @staticmethod
    def aliases(directory: Path, stem: str, real_name: str):
        major_name = stem + ".1"
        os.symlink(real_name, directory / major_name)
        os.symlink(major_name, directory / stem)

    def _build_default_libraries(self):
        leaf_source = self.source(
            "leaf.c",
            """
            #include <stdio.h>
            int m4_leaf(void) { return puts("fixture library is not executed"); }
            """,
        )
        leaf = self.libdir / "libm4leaf.so.1.0"
        self.shared(leaf, "libm4leaf.so.1", leaf_source)
        self.aliases(self.libdir, "libm4leaf.so", leaf.name)

        dep_source = self.source(
            "dep.c",
            """
            extern int m4_leaf(void);
            int m4_dep(void) { return m4_leaf() + 1; }
            """,
        )
        dep = self.libdir / "libm4dep.so.1.0"
        self.shared(
            dep,
            "libm4dep.so.1",
            dep_source,
            library_dirs=(self.libdir,),
            libraries=("m4leaf",),
        )
        self.aliases(self.libdir, "libm4dep.so", dep.name)

        self._build_cuda_libraries()

        gluten_source = self.source(
            "gluten.c",
            """
            extern int m4_dep(void);
            extern int cudart_fixture(void);
            int gluten_fixture(void) { return m4_dep() + cudart_fixture(); }
            """,
        )
        self.libgluten = self.root / "libgluten.so"
        self.shared(
            self.libgluten,
            "libgluten.so",
            gluten_source,
            library_dirs=(self.libdir,),
            libraries=("m4dep",),
        )

        uct_source = self.source(
            "uct_cuda.c",
            """
            extern int m4_dep(void);
            int uct_cuda_fixture(void) { return m4_dep(); }
            """,
        )
        uct = self.ucx / "libuct_cuda.so.1.0"
        self.shared(
            uct,
            "libuct_cuda.so.1",
            uct_source,
            library_dirs=(self.libdir,),
            libraries=("m4dep",),
        )
        self.aliases(self.ucx, "libuct_cuda.so", uct.name)

        ucm_source = self.source(
            "ucm_cuda.c",
            """
            int ucm_cuda_fixture(void) { return 3; }
            """,
        )
        ucm = self.ucx / "libucm_cuda.so.1.0"
        self.shared(ucm, "libucm_cuda.so.1", ucm_source)
        self.aliases(self.ucx, "libucm_cuda.so", ucm.name)

    def _cuda_family(self, stem: str, soname: str, real_name: str, symbol: str) -> Path:
        source = self.source(
            f"{symbol}.c",
            f"""
            int {symbol}(void) {{ return 5; }}
            """,
        )
        real = self.cuda / real_name
        self.shared(real, soname, source)
        os.symlink(real.name, self.cuda / soname)
        os.symlink(soname, self.cuda / stem)
        return real

    def _build_cuda_libraries(self):
        self._cuda_family(
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.12.4.99",
            "cudart_fixture",
        )
        self._cuda_family(
            "libnvrtc.so",
            "libnvrtc.so.12",
            "libnvrtc.so.12.4.99",
            "nvrtc_fixture",
        )
        self._cuda_family(
            "libnvrtc-builtins.so",
            "libnvrtc-builtins.so.12.4",
            "libnvrtc-builtins.so.12.4.99",
            "nvrtc_builtins_fixture",
        )
        self._cuda_family(
            "libnvJitLink.so",
            "libnvJitLink.so.12",
            "libnvJitLink.so.12.4.99",
            "nvjitlink_fixture",
        )

    def build(self, *, native_only: bool = False):
        return packager.build_bundle(
            None if native_only else self.jar,
            self.libgluten,
            self.output,
            gluten_revision=GLUTEN_REVISION,
            velox_revision=VELOX_REVISION,
            ucx_dirs=[self.ucx],
        )


class PackagerTest(unittest.TestCase):
    def fixture(self, root: Path) -> Fixture:
        return Fixture(root)

    def assert_rtcx_bundle_content(self, fixture: Fixture):
        libs = fixture.output / "libs"
        families = (
            ("libcudart.so", "libcudart.so.12", "libcudart.so.12.4.99"),
            ("libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.12.4.99"),
            (
                "libnvrtc-builtins.so",
                "libnvrtc-builtins.so.12.4",
                "libnvrtc-builtins.so.12.4.99",
            ),
            ("libnvJitLink.so", "libnvJitLink.so.12", "libnvJitLink.so.12.4.99"),
        )
        for unversioned, soname, real_name in families:
            with self.subTest(family=unversioned):
                self.assertTrue((libs / real_name).is_file())
                self.assertEqual(soname, os.readlink(libs / unversioned))
                self.assertEqual(real_name, os.readlink(libs / soname))
                self.assertEqual(
                    "$ORIGIN",
                    subprocess.check_output(
                        ["patchelf", "--print-rpath", libs / real_name], text=True
                    ).strip(),
                )

    def test_libxcrypt_is_bundle_content(self):
        self.assertIsNone(packager._external_provider_kind("libcrypt.so.1"))

    def test_ldd_parser_records_direct_dynamic_loader_path(self):
        resolutions = packager._parse_ldd_resolutions("""
            libgcc_s.so.1 => /usr/lib64/libgcc_s.so.1 (0x0001)
            libmissing.so.1 => not found
            /lib64/ld-linux-x86-64.so.2 (0x0002)
            """)

        self.assertEqual(Path("/usr/lib64/libgcc_s.so.1"), resolutions["libgcc_s.so.1"])
        self.assertIsNone(resolutions["libmissing.so.1"])
        self.assertEqual(
            Path("/lib64/ld-linux-x86-64.so.2"),
            resolutions["ld-linux-x86-64.so.2"],
        )

    def test_preserves_soname_alias_when_real_filename_has_a_vendor_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            source = fixture.source(
                "vendor-suffix.c",
                """
                int m4_vendor_suffix(void) { return 17; }
                """,
            )
            real = fixture.libdir / "libm4vendor-build.so.1"
            fixture.shared(real, "libm4vendor.so.1", source)
            os.symlink(real.name, fixture.libdir / "libm4vendor.so.1")
            os.symlink("libm4vendor.so.1", fixture.libdir / "libm4vendor.so")

            gluten_source = fixture.source(
                "gluten-vendor-suffix.c",
                """
                extern int m4_vendor_suffix(void);
                int gluten_fixture(void) { return m4_vendor_suffix(); }
                """,
            )
            fixture.shared(
                fixture.libgluten,
                "libgluten.so",
                gluten_source,
                library_dirs=(fixture.libdir,),
                libraries=("m4vendor",),
            )

            fixture.build()

            libs = fixture.output / "libs"
            self.assertTrue((libs / real.name).is_file())
            self.assertEqual(real.name, os.readlink(libs / "libm4vendor.so.1"))
            self.assertEqual("libm4vendor.so.1", os.readlink(libs / "libm4vendor.so"))

    def test_recursive_closure_aliases_ucx_rpath_and_exact_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.build()

            self.assertEqual(
                {fixture.jar.name, "libs"},
                {path.name for path in fixture.output.iterdir()},
            )
            libs = fixture.output / "libs"
            ucx = libs / "ucx"
            self.assertEqual(
                packager.NativeBuildInfo(GLUTEN_REVISION, VELOX_REVISION),
                packager.read_native_build_info(libs / packager.NATIVE_BUILD_INFO_NAME),
            )
            self.assertTrue((libs / "libgluten.so").is_file())
            self.assertTrue((libs / "libm4dep.so.1.0").is_file())
            self.assertTrue((libs / "libm4leaf.so.1.0").is_file())
            self.assertEqual("libm4dep.so.1", os.readlink(libs / "libm4dep.so"))
            self.assertEqual("libm4dep.so.1.0", os.readlink(libs / "libm4dep.so.1"))
            self.assertTrue((ucx / "libuct_cuda.so.1.0").is_file())
            self.assertTrue((ucx / "libucm_cuda.so.1.0").is_file())
            self.assertEqual(
                "$ORIGIN",
                subprocess.check_output(
                    ["patchelf", "--print-rpath", libs / "libgluten.so"], text=True
                ).strip(),
            )
            self.assertEqual(
                "$ORIGIN:$ORIGIN/..",
                subprocess.check_output(
                    ["patchelf", "--print-rpath", ucx / "libuct_cuda.so.1.0"],
                    text=True,
                ).strip(),
            )
            self.assert_rtcx_bundle_content(fixture)

    def test_native_only_build_needs_no_jar_and_keeps_native_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.jar.unlink()
            fixture.build(native_only=True)

            self.assertEqual(
                {"libs"},
                {path.name for path in fixture.output.iterdir()},
            )
            libs = fixture.output / "libs"
            ucx = libs / "ucx"
            self.assertEqual(
                "backend_type=velox\n"
                f"gluten_revision={GLUTEN_REVISION}\n"
                f"velox_revision={VELOX_REVISION}\n",
                (libs / packager.NATIVE_BUILD_INFO_NAME).read_text(encoding="utf-8"),
            )
            self.assertTrue((libs / "libgluten.so").is_file())
            self.assertTrue((libs / "libm4dep.so.1.0").is_file())
            self.assertEqual("libm4dep.so.1", os.readlink(libs / "libm4dep.so"))
            self.assertTrue((ucx / "libuct_cuda.so.1.0").is_file())
            self.assertTrue((ucx / "libucm_cuda.so.1.0").is_file())
            self.assertEqual(
                "$ORIGIN",
                subprocess.check_output(
                    ["patchelf", "--print-rpath", libs / "libgluten.so"], text=True
                ).strip(),
            )
            self.assertEqual(
                "$ORIGIN:$ORIGIN/..",
                subprocess.check_output(
                    ["patchelf", "--print-rpath", ucx / "libuct_cuda.so.1.0"],
                    text=True,
                ).strip(),
            )
            self.assert_rtcx_bundle_content(fixture)

    def test_composes_validated_native_tree_without_native_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.build(native_only=True)
            composed = fixture.root / "composed"
            composed.mkdir()
            source_alias = os.readlink(fixture.output / "libs" / "libm4dep.so")

            with mock.patch.object(
                packager,
                "_discover_ucx_dirs",
                side_effect=AssertionError("composition ran discovery"),
            ), mock.patch.object(
                packager,
                "_plan_objects",
                side_effect=AssertionError("composition replanned native closure"),
            ), mock.patch.object(
                packager,
                "_copy_and_patch",
                side_effect=AssertionError("composition repatched native closure"),
            ):
                packager.compose_bundle(fixture.output, fixture.jar, composed)

            self.assertEqual(
                {fixture.jar.name, "libs"},
                {path.name for path in composed.iterdir()},
            )
            self.assertEqual(
                source_alias, os.readlink(composed / "libs" / "libm4dep.so")
            )
            self.assertEqual(
                packager.NativeBuildInfo(GLUTEN_REVISION, VELOX_REVISION),
                packager.read_native_build_info(
                    composed / "libs" / packager.NATIVE_BUILD_INFO_NAME
                ),
            )

    def test_accepts_source_supported_spark4_vendor_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.jar.unlink()
            fixture.jar = fixture.root / (
                "gluten-velox-bundle-spark4.0_2.13-1.6.0-SNAPSHOT.jar"
            )
            fixture.write_jar(
                fixture.jar,
                spark_version="4.0.2-amzn-0",
                scala_version="2.13.16",
            )

            fixture.build()

            self.assertTrue((fixture.output / fixture.jar.name).is_file())

    def test_rejects_direct_and_composed_revision_mismatches(self):
        for field in ("gluten", "velox"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                fixture.write_jar(
                    fixture.jar,
                    gluten_revision="3" * 40 if field == "gluten" else GLUTEN_REVISION,
                    velox_revision="4" * 40 if field == "velox" else VELOX_REVISION,
                )
                with self.assertRaisesRegex(packager.PackagerError, "do not match"):
                    fixture.build()
                self.assertEqual([], list(fixture.output.iterdir()))

                fixture.write_jar(fixture.jar)
                fixture.build(native_only=True)
                fixture.write_jar(
                    fixture.jar,
                    gluten_revision="3" * 40 if field == "gluten" else GLUTEN_REVISION,
                    velox_revision="4" * 40 if field == "velox" else VELOX_REVISION,
                )
                composed = fixture.root / "composed"
                composed.mkdir()
                with self.assertRaisesRegex(packager.PackagerError, "do not match"):
                    packager.compose_bundle(fixture.output, fixture.jar, composed)
                self.assertEqual([], list(composed.iterdir()))

    def test_rejects_crossed_and_unsupported_jvm_artifacts(self):
        cases = (
            (
                "gluten-velox-bundle-spark3.5_2.12-crossed.jar",
                "4.0.0",
                "2.13.16",
                "filename Spark line",
            ),
            (
                "gluten-velox-bundle-spark9.9_2.12-unsupported.jar",
                "9.9.0",
                "2.12.15",
                "no source shim profile",
            ),
        )
        for name, spark_version, scala_version, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                artifact = fixture.root / name
                fixture.write_jar(
                    artifact,
                    spark_version=spark_version,
                    scala_version=scala_version,
                )
                with self.assertRaisesRegex(packager.PackagerError, expected):
                    packager._validate_jar(artifact)

    def test_rejects_duplicate_selected_shim_provider_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            provider = "org/apache/gluten/sql/shims/spark35/SparkShimProvider.class"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(fixture.jar, "a") as archive:
                    archive.writestr(provider, b"duplicate")
            with self.assertRaisesRegex(
                packager.PackagerError, "exactly its selected Spark shim provider"
            ):
                packager._validate_jar(fixture.jar)

    def test_composition_rejects_tampered_metadata_layout_and_links(self):
        cases = (
            "metadata",
            "extra-directory",
            "absolute-link",
            "non-elf-link",
            "spoofed-ucx",
            "crossed-needed-alias",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                fixture.build(native_only=True)
                if case == "metadata":
                    (
                        fixture.output / "libs" / packager.NATIVE_BUILD_INFO_NAME
                    ).write_text("backend_type=velox\n", encoding="utf-8")
                elif case == "extra-directory":
                    (fixture.output / "libs" / "unexpected").mkdir()
                elif case == "absolute-link":
                    (fixture.output / "libs" / "unsafe.so").symlink_to(
                        "/usr/lib/libunsafe.so"
                    )
                elif case == "non-elf-link":
                    (fixture.output / "libs" / "not-a-library.so").symlink_to(
                        packager.NATIVE_BUILD_INFO_NAME
                    )
                elif case == "spoofed-ucx":
                    ucx = fixture.output / "libs" / "ucx"
                    for stem in ("libuct_cuda.so", "libucm_cuda.so"):
                        for path in ucx.glob(f"{stem}*"):
                            path.unlink()
                        (ucx / stem).symlink_to("../libcudart.so.12")
                else:
                    alias = fixture.output / "libs" / "libm4dep.so.1"
                    alias.unlink()
                    alias.symlink_to("libcudart.so.12")
                composed = fixture.root / "composed"
                composed.mkdir()
                with self.assertRaises(packager.PackagerError):
                    packager.compose_bundle(fixture.output, fixture.jar, composed)
                self.assertEqual([], list(composed.iterdir()))

    def test_cli_requires_exactly_one_full_or_native_only_mode(self):
        common = [
            "--libgluten=/tmp/libgluten.so",
            f"--gluten_revision={GLUTEN_REVISION}",
            f"--velox_revision={VELOX_REVISION}",
            "--output_dir=/tmp/deploy",
        ]
        full = packager._parse_args(["--bundle_jar=/tmp/bundle.jar", *common])
        native = packager._parse_args(["--native_only", *common])
        self.assertEqual(Path("/tmp/bundle.jar"), full.bundle_jar)
        self.assertFalse(full.native_only)
        self.assertIsNone(native.bundle_jar)
        self.assertTrue(native.native_only)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                packager._parse_args(common)
            with self.assertRaises(SystemExit):
                packager._parse_args(
                    ["--native_only", "--bundle_jar=/tmp/bundle.jar", *common]
                )
            with self.assertRaises(SystemExit):
                packager._parse_args(
                    [
                        "--native_bundle=/tmp/native",
                        "--native_only",
                        "--output_dir=/tmp/deploy",
                    ]
                )

        composed = packager._parse_args(
            [
                "--native_bundle=/tmp/native",
                "--bundle_jar=/tmp/bundle.jar",
                "--output_dir=/tmp/deploy",
            ]
        )
        self.assertEqual(Path("/tmp/native"), composed.native_bundle)

    def test_native_only_tool_preflight_does_not_require_jar(self):
        available = {"ldd", "patchelf", "readelf"}
        with mock.patch.object(
            packager.shutil,
            "which",
            side_effect=lambda command: (
                f"/usr/bin/{command}" if command in available else None
            ),
        ):
            packager._preflight_tools(require_jar=False)
            with self.assertRaisesRegex(packager.PackagerError, "jar"):
                packager._preflight_tools(require_jar=True)

    def test_rejects_unsafe_ucx_symlinks(self):
        cases = ("broken", "absolute", "escaping", "cyclic")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                if case == "broken":
                    os.symlink("missing.so", fixture.ucx / "libuct_cuda.so.broken")
                elif case == "absolute":
                    os.symlink(
                        str(fixture.ucx / "libuct_cuda.so.1.0"),
                        fixture.ucx / "libuct_cuda.so.absolute",
                    )
                elif case == "escaping":
                    outside = fixture.root / "outside"
                    outside.mkdir()
                    shutil.copy2(
                        fixture.ucx / "libuct_cuda.so.1.0",
                        outside / "libuct_cuda.so.1.0",
                    )
                    os.symlink(
                        "../outside/libuct_cuda.so.1.0",
                        fixture.ucx / "libuct_cuda.so.escape",
                    )
                else:
                    os.symlink(
                        "libuct_cuda.so.cycle-b", fixture.ucx / "libuct_cuda.so.cycle-a"
                    )
                    os.symlink(
                        "libuct_cuda.so.cycle-a", fixture.ucx / "libuct_cuda.so.cycle-b"
                    )
                with self.assertRaises(packager.PackagerError):
                    fixture.build()
                self.assertEqual([], list(fixture.output.iterdir()))

    def test_rejects_missing_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            for name in ("libm4leaf.so", "libm4leaf.so.1", "libm4leaf.so.1.0"):
                (fixture.libdir / name).unlink()
            with self.assertRaisesRegex(
                packager.PackagerError, "unresolved dependency"
            ):
                fixture.build()

    def test_selects_one_soname_family_across_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            first = fixture.root / "collision-a"
            second = fixture.root / "collision-b"
            first.mkdir()
            second.mkdir()
            same_source = fixture.source(
                "same.c",
                """
                int m4_same(void) { return 11; }
                """,
            )
            first_real = first / "libm4same-first.so.1"
            second_real = second / "libm4same-second.so.1"
            for directory, real in ((first, first_real), (second, second_real)):
                fixture.shared(real, "libm4same.so.1", same_source)
                fixture.aliases(directory, "libm4same.so", real.name)

            gluten_source = fixture.source(
                "gluten-collision.c",
                """
                extern int m4_same(void);
                int gluten_fixture(void) { return m4_same(); }
                """,
            )
            fixture.shared(
                fixture.libgluten,
                "libgluten.so",
                gluten_source,
                library_dirs=(first,),
                libraries=("m4same",),
            )
            uct_source = fixture.source(
                "uct-collision.c",
                """
                extern int m4_same(void);
                int uct_cuda_fixture(void) { return m4_same(); }
                """,
            )
            fixture.shared(
                fixture.ucx / "libuct_cuda.so.1.0",
                "libuct_cuda.so.1",
                uct_source,
                library_dirs=(second,),
                libraries=("m4same",),
            )
            fixture.build()

            libs = fixture.output / "libs"
            self.assertTrue((libs / first_real.name).is_file())
            self.assertFalse((libs / second_real.name).exists())
            self.assertEqual(first_real.name, os.readlink(libs / "libm4same.so.1"))

    def test_libgluten_transitive_soname_precedes_ucx_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            first = fixture.root / "choice-a"
            second = fixture.root / "choice-b"
            first.mkdir()
            second.mkdir()
            choice_source = fixture.source(
                "choice.c",
                """
                int m4_choice(void) { return 13; }
                """,
            )
            first_real = first / "libm4choice-first.so.1"
            second_real = second / "libm4choice-second.so.1"
            for directory, real in ((first, first_real), (second, second_real)):
                fixture.shared(real, "libm4choice.so.1", choice_source)
                fixture.aliases(directory, "libm4choice.so", real.name)

            mid_source = fixture.source(
                "mid.c",
                """
                extern int m4_choice(void);
                int m4_mid(void) { return m4_choice(); }
                """,
            )
            mid = fixture.libdir / "libm4mid.so.1.0"
            fixture.shared(
                mid,
                "libm4mid.so.1",
                mid_source,
                library_dirs=(first,),
                libraries=("m4choice",),
            )
            fixture.aliases(fixture.libdir, "libm4mid.so", mid.name)

            gluten_source = fixture.source(
                "gluten-transitive.c",
                """
                extern int m4_mid(void);
                int gluten_fixture(void) { return m4_mid(); }
                """,
            )
            fixture.shared(
                fixture.libgluten,
                "libgluten.so",
                gluten_source,
                library_dirs=(fixture.libdir,),
                libraries=("m4mid",),
            )
            uct_source = fixture.source(
                "uct-transitive.c",
                """
                extern int m4_choice(void);
                int uct_cuda_fixture(void) { return m4_choice(); }
                """,
            )
            fixture.shared(
                fixture.ucx / "libuct_cuda.so.1.0",
                "libuct_cuda.so.1",
                uct_source,
                library_dirs=(second,),
                libraries=("m4choice",),
            )

            fixture.build()

            libs = fixture.output / "libs"
            self.assertTrue((libs / first_real.name).is_file())
            self.assertFalse((libs / second_real.name).exists())
            self.assertEqual(first_real.name, os.readlink(libs / "libm4choice.so.1"))

    def test_rejects_real_filename_collision_between_different_sonames(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            first = fixture.root / "collision-a"
            second = fixture.root / "collision-b"
            first.mkdir()
            second.mkdir()
            first_source = fixture.source(
                "first.c",
                """
                int m4_first(void) { return 11; }
                """,
            )
            second_source = fixture.source(
                "second.c",
                """
                int m4_second(void) { return 12; }
                """,
            )
            real_name = "libm4collision.so.1.0"
            fixture.shared(first / real_name, "libm4first.so.1", first_source)
            fixture.aliases(first, "libm4first.so", real_name)
            fixture.shared(second / real_name, "libm4second.so.1", second_source)
            fixture.aliases(second, "libm4second.so", real_name)

            gluten_source = fixture.source(
                "gluten-collision.c",
                """
                extern int m4_first(void);
                int gluten_fixture(void) { return m4_first(); }
                """,
            )
            fixture.shared(
                fixture.libgluten,
                "libgluten.so",
                gluten_source,
                library_dirs=(first,),
                libraries=("m4first",),
            )
            uct_source = fixture.source(
                "uct-collision.c",
                """
                extern int m4_second(void);
                int uct_cuda_fixture(void) { return m4_second(); }
                """,
            )
            fixture.shared(
                fixture.ucx / "libuct_cuda.so.1.0",
                "libuct_cuda.so.1",
                uct_source,
                library_dirs=(second,),
                libraries=("m4second",),
            )

            with self.assertRaisesRegex(packager.PackagerError, "basename collision"):
                fixture.build()

    def test_real_basename_does_not_masquerade_as_a_selected_soname(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            first = fixture.root / "collision-a"
            second = fixture.root / "collision-b"
            first.mkdir()
            second.mkdir()
            first_source = fixture.source(
                "first-real-name.c",
                """
                int m4_first(void) { return 21; }
                """,
            )
            first_real = first / "libm4collision.so.1"
            fixture.shared(first_real, "libm4first.so.1", first_source)
            fixture.aliases(first, "libm4first.so", first_real.name)

            second_source = fixture.source(
                "second-soname.c",
                """
                int m4_second(void) { return 22; }
                """,
            )
            second_real = second / "libm4second.so.1.0"
            fixture.shared(second_real, "libm4collision.so.1", second_source)
            fixture.aliases(second, "libm4collision.so", second_real.name)

            gluten_source = fixture.source(
                "gluten-real-name.c",
                """
                extern int m4_first(void);
                int gluten_fixture(void) { return m4_first(); }
                """,
            )
            fixture.shared(
                fixture.libgluten,
                "libgluten.so",
                gluten_source,
                library_dirs=(first,),
                libraries=("m4first",),
            )
            uct_source = fixture.source(
                "uct-second-soname.c",
                """
                extern int m4_second(void);
                int uct_cuda_fixture(void) { return m4_second(); }
                """,
            )
            fixture.shared(
                fixture.ucx / "libuct_cuda.so.1.0",
                "libuct_cuda.so.1",
                uct_source,
                library_dirs=(second,),
                libraries=("m4collision",),
            )

            with self.assertRaisesRegex(packager.PackagerError, "basename collision"):
                fixture.build()

    def test_rejects_missing_cuda_module_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            for name in ("libucm_cuda.so", "libucm_cuda.so.1", "libucm_cuda.so.1.0"):
                (fixture.ucx / name).unlink()
            with self.assertRaisesRegex(packager.PackagerError, "libucm_cuda"):
                fixture.build()

    def test_rejects_each_missing_rtcx_provider_family(self):
        for stem in packager.RTCX_FAMILY_STEMS:
            with self.subTest(family=stem), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                for path in fixture.cuda.glob(f"{stem}*"):
                    path.unlink()
                with self.assertRaisesRegex(packager.PackagerError, stem):
                    fixture.build()
                self.assertEqual([], list(fixture.output.iterdir()))

    def test_rtcx_discovery_does_not_fall_back_to_another_toolkit(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            decoy = fixture.root / "cuda-decoy" / "lib"
            shutil.copytree(fixture.cuda, decoy, symlinks=True)
            for path in fixture.cuda.glob("libnvrtc.so*"):
                path.unlink()

            with self.assertRaisesRegex(packager.PackagerError, "libnvrtc[.]so"):
                fixture.build()
            self.assertEqual([], list(fixture.output.iterdir()))

    def test_rejects_rtcx_provider_from_a_different_cuda_major(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            for path in fixture.cuda.glob("libnvJitLink.so*"):
                path.unlink()
            fixture._cuda_family(
                "libnvJitLink.so",
                "libnvJitLink.so.11",
                "libnvJitLink.so.11.8.99",
                "nvjitlink_wrong_major",
            )

            with self.assertRaisesRegex(packager.PackagerError, "CUDA major 12"):
                fixture.build()
            self.assertEqual([], list(fixture.output.iterdir()))

    def test_rejects_multiple_real_files_in_one_rtcx_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            source = fixture.source(
                "second-nvrtc.c",
                """
                int second_nvrtc_fixture(void) { return 9; }
                """,
            )
            fixture.shared(
                fixture.cuda / "libnvrtc.so.12.5.77",
                "libnvrtc.so.12",
                source,
            )

            with self.assertRaisesRegex(packager.PackagerError, "multiple real files"):
                fixture.build()
            self.assertEqual([], list(fixture.output.iterdir()))

    def test_output_shape_rejects_zero_or_two_jars(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.build()
            copied_jar = fixture.output / fixture.jar.name
            copied_jar.unlink()
            with self.assertRaisesRegex(packager.PackagerError, "exactly one"):
                packager._validate_output_shape(fixture.output)

            shutil.copy2(fixture.jar, copied_jar)
            shutil.copy2(
                fixture.jar,
                fixture.output / "gluten-velox-bundle-spark3.5_2.12-second.jar",
            )
            with self.assertRaisesRegex(packager.PackagerError, "exactly one"):
                packager._validate_output_shape(fixture.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
