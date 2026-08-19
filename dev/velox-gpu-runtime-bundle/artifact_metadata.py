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

"""Read the minimal identity shared by native and JVM bundle artifacts.

The runtime packager can produce its native closure before a Maven bundle is
built.  These helpers keep that asynchronous join safe without introducing a
receipt format: the native tree carries three plain properties, while the JVM
artifact continues to use Gluten's existing ``gluten-build-info.properties``.
The canonical JAR name and current source POM remain the authority for the
Spark shim and Scala pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

NATIVE_BUILD_INFO_NAME = "gluten-native-build-info.properties"
JAR_BUILD_INFO_NAME = "gluten-build-info.properties"
SHIM_SERVICE_NAME = "META-INF/services/org.apache.gluten.sql.shims.SparkShimProvider"
FULL_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
CANONICAL_BUNDLE_JAR_RE = re.compile(
    r"gluten-velox-bundle-spark"
    r"(?P<spark_line>[0-9]+[.][0-9]+)_"
    r"(?P<scala_binary>[0-9]+[.][0-9]+)-"
    r"(?P<suffix>[^/]+)[.]jar\Z"
)
EXACT_SPARK_VERSION_RE = re.compile(
    r"(?P<spark_line>[0-9]+[.][0-9]+)[.][0-9]+" r"(?:[-.][A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)
SCALA_VERSION_RE = re.compile(
    r"(?P<scala_binary>[0-9]+[.][0-9]+)(?:[.][0-9]+)?"
    r"(?:[-.][A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)


class ArtifactMetadataError(RuntimeError):
    """An artifact name, properties file, or source profile is inconsistent."""


@dataclass(frozen=True)
class NativeBuildInfo:
    gluten_revision: str
    velox_revision: str
    backend_type: str = "velox"


@dataclass(frozen=True)
class BundleJarInfo:
    path: Path
    spark_line: str
    scala_binary: str
    spark_version: str
    scala_version: str
    java_version: str
    gluten_revision: str
    velox_revision: str
    backend_type: str


@dataclass(frozen=True)
class SourceProfile:
    profile: str
    spark_line: str
    default_spark_version: str
    scala_binary: str
    allowed_scala_binaries: tuple[str, ...]


def _readable_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ArtifactMetadataError(
            f"{label} must be a real file, not a symlink: {path}"
        )
    try:
        mode = path.stat().st_mode
        size = path.stat().st_size
    except OSError as error:
        raise ArtifactMetadataError(
            f"{label} is missing or inaccessible: {path}"
        ) from error
    if not stat.S_ISREG(mode) or not os.access(path, os.R_OK) or size == 0:
        raise ArtifactMetadataError(f"{label} must be a readable nonempty file: {path}")
    return path.parent.resolve(strict=True) / path.name


def parse_properties(text: str, label: str) -> dict[str, str]:
    """Parse the simple ``key=value`` form emitted by Gluten's build scripts."""

    values: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if "=" not in line:
            raise ArtifactMetadataError(
                f"{label} line {number} is not a key=value property"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ArtifactMetadataError(f"{label} line {number} has an empty key")
        if key in values:
            raise ArtifactMetadataError(f"{label} repeats property {key}")
        values[key] = value
    return values


def _require_revision(value: str, field: str, label: str) -> str:
    if not FULL_REVISION_RE.fullmatch(value):
        raise ArtifactMetadataError(
            f"{label} property {field} must be one full lowercase Git revision"
        )
    return value


def native_build_info(gluten_revision: str, velox_revision: str) -> NativeBuildInfo:
    return NativeBuildInfo(
        gluten_revision=_require_revision(
            gluten_revision, "gluten_revision", "native build info"
        ),
        velox_revision=_require_revision(
            velox_revision, "velox_revision", "native build info"
        ),
    )


def write_native_build_info(path: Path, info: NativeBuildInfo) -> None:
    path.write_text(
        "backend_type=velox\n"
        f"gluten_revision={info.gluten_revision}\n"
        f"velox_revision={info.velox_revision}\n",
        encoding="utf-8",
    )


def read_native_build_info(path: Path) -> NativeBuildInfo:
    source = _readable_regular_file(path, "native build info")
    try:
        values = parse_properties(source.read_text(encoding="utf-8"), str(source))
    except (OSError, UnicodeError) as error:
        raise ArtifactMetadataError(
            f"could not read native build info: {source}"
        ) from error
    expected = {"backend_type", "gluten_revision", "velox_revision"}
    if set(values) != expected:
        raise ArtifactMetadataError(
            "native build info must contain exactly backend_type, "
            "gluten_revision, and velox_revision"
        )
    if values["backend_type"] != "velox":
        raise ArtifactMetadataError("native build info backend_type must be velox")
    return native_build_info(values["gluten_revision"], values["velox_revision"])


def exact_spark_line(version: str, label: str = "spark_version") -> str:
    match = EXACT_SPARK_VERSION_RE.fullmatch(version)
    if match is None:
        raise ArtifactMetadataError(
            f"{label} must be an exact Spark X.Y.Z coordinate with an optional vendor suffix"
        )
    return match.group("spark_line")


def scala_binary_version(version: str, label: str = "scala_version") -> str:
    match = SCALA_VERSION_RE.fullmatch(version)
    if match is None:
        raise ArtifactMetadataError(f"{label} is not a supported Scala version")
    return match.group("scala_binary")


def _required_property(values: dict[str, str], key: str, label: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ArtifactMetadataError(f"{label} is missing nonempty property {key}")
    return value


def read_bundle_jar(path: Path) -> BundleJarInfo:
    source = _readable_regular_file(path, "bundle JAR")
    name_match = CANONICAL_BUNDLE_JAR_RE.fullmatch(source.name)
    if name_match is None:
        raise ArtifactMetadataError(
            "bundle JAR name must match " "gluten-velox-bundle-sparkX.Y_A.B-*.jar"
        )
    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            matches = [name for name in names if name == JAR_BUILD_INFO_NAME]
            if len(matches) != 1:
                raise ArtifactMetadataError(
                    "bundle JAR must contain exactly one gluten-build-info.properties"
                )
            raw_info = archive.read(matches[0]).decode("utf-8")
            service_matches = [name for name in names if name == SHIM_SERVICE_NAME]
            if len(service_matches) != 1:
                raise ArtifactMetadataError(
                    "bundle JAR must contain exactly one SparkShimProvider service entry"
                )
            raw_service = archive.read(service_matches[0]).decode("utf-8")
            provider_classes = [
                name
                for name in names
                if re.fullmatch(
                    r"org/apache/gluten/sql/shims/spark[0-9]+/SparkShimProvider[.]class",
                    name,
                )
            ]
            if archive.testzip() is not None:
                raise ArtifactMetadataError(
                    f"bundle JAR has an unreadable entry: {source}"
                )
    except ArtifactMetadataError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as error:
        raise ArtifactMetadataError(
            f"bundle JAR is not a readable ZIP: {source}"
        ) from error

    label = f"{source}:{JAR_BUILD_INFO_NAME}"
    values = parse_properties(raw_info, label)
    backend = _required_property(values, "backend_type", label)
    if backend != "velox":
        raise ArtifactMetadataError("bundle JAR backend_type must be velox")
    spark_version = _required_property(values, "spark_version", label)
    scala_version = _required_property(values, "scala_version", label)
    java_version = _required_property(values, "java_version", label)
    if java_version != "17":
        raise ArtifactMetadataError("bundle JAR java_version must be 17")
    spark_line = exact_spark_line(spark_version)
    scala_binary = scala_binary_version(scala_version)
    if spark_line != name_match.group("spark_line"):
        raise ArtifactMetadataError(
            "bundle JAR filename Spark line does not match embedded spark_version"
        )
    if scala_binary != name_match.group("scala_binary"):
        raise ArtifactMetadataError(
            "bundle JAR filename Scala binary does not match embedded scala_version"
        )
    shim_suffix = spark_line.replace(".", "")
    expected_provider = (
        f"org.apache.gluten.sql.shims.spark{shim_suffix}.SparkShimProvider"
    )
    expected_class = expected_provider.replace(".", "/") + ".class"
    if provider_classes != [expected_class]:
        raise ArtifactMetadataError(
            "bundle JAR does not contain exactly its selected Spark shim provider class"
        )
    service_providers = [
        line.strip()
        for line in raw_service.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if service_providers != [expected_provider]:
        raise ArtifactMetadataError(
            "bundle JAR SparkShimProvider service does not match its selected Spark line"
        )
    return BundleJarInfo(
        path=source,
        spark_line=spark_line,
        scala_binary=scala_binary,
        spark_version=spark_version,
        scala_version=scala_version,
        java_version=java_version,
        gluten_revision=_require_revision(
            _required_property(values, "revision", label), "revision", label
        ),
        velox_revision=_require_revision(
            _required_property(values, "velox_revision", label),
            "velox_revision",
            label,
        ),
        backend_type=backend,
    )


def source_profiles(pom_path: Path) -> dict[str, SourceProfile]:
    source = _readable_regular_file(pom_path, "root pom.xml")
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise ArtifactMetadataError(
            f"could not parse root pom.xml: {source}"
        ) from error
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    root_properties_node = root.find("m:properties", namespace)
    root_properties = (
        {
            child.tag.rsplit("}", 1)[-1]: (child.text or "").strip()
            for child in root_properties_node
        }
        if root_properties_node is not None
        else {}
    )
    default_scala = root_properties.get("scala.binary.version", "")
    declared_scala_binaries = {
        match.group(1)
        for profile_node in root.findall("m:profiles/m:profile", namespace)
        for profile_id in (
            profile_node.findtext("m:id", default="", namespaces=namespace).strip(),
        )
        for match in (re.fullmatch(r"scala-([0-9]+[.][0-9]+)", profile_id),)
        if match is not None
    }
    if not declared_scala_binaries:
        raise ArtifactMetadataError("root pom.xml has no declared Scala profiles")
    profiles: dict[str, SourceProfile] = {}
    for profile_node in root.findall("m:profiles/m:profile", namespace):
        profile_id = (
            profile_node.findtext("m:id", default="", namespaces=namespace)
        ).strip()
        match = re.fullmatch(r"spark-([0-9]+[.][0-9]+)", profile_id)
        if match is None:
            continue
        properties_node = profile_node.find("m:properties", namespace)
        properties = (
            {
                child.tag.rsplit("}", 1)[-1]: (child.text or "").strip()
                for child in properties_node
            }
            if properties_node is not None
            else {}
        )
        spark_line = match.group(1)
        if properties.get("sparkbundle.version") != spark_line:
            raise ArtifactMetadataError(
                f"source profile {profile_id} has inconsistent sparkbundle.version"
            )
        if not properties.get("sparkshim.artifactId"):
            raise ArtifactMetadataError(
                f"source profile {profile_id} has no sparkshim.artifactId"
            )
        default_spark_version = properties.get("spark.version", "")
        if (
            exact_spark_line(default_spark_version, f"{profile_id} spark.version")
            != spark_line
        ):
            raise ArtifactMetadataError(
                f"source profile {profile_id} has an inconsistent spark.version"
            )
        required_scala_profiles = {
            value.removeprefix("scala-")
            for node in profile_node.findall(
                ".//m:requireActiveProfile/m:profiles", namespace
            )
            for raw_value in (node.text or "").split(",")
            for value in (raw_value.strip(),)
            if value.startswith("scala-")
        }
        if len(required_scala_profiles) > 1:
            raise ArtifactMetadataError(
                f"source profile {profile_id} requires multiple Scala profiles"
            )
        required_scala = next(iter(required_scala_profiles), "")
        constrained_scala = properties.get("scala.binary.version", required_scala)
        scala_binary = constrained_scala or default_scala
        if required_scala and scala_binary != required_scala:
            raise ArtifactMetadataError(
                f"source profile {profile_id} has conflicting Scala requirements"
            )
        if not re.fullmatch(r"[0-9]+[.][0-9]+", scala_binary):
            raise ArtifactMetadataError(
                f"source profile {profile_id} has no valid Scala binary version"
            )
        allowed_scala_binaries = (
            (constrained_scala,)
            if constrained_scala
            else tuple(sorted(declared_scala_binaries))
        )
        if scala_binary not in allowed_scala_binaries:
            raise ArtifactMetadataError(
                f"source profile {profile_id} defaults to undeclared Scala {scala_binary}"
            )
        profiles[profile_id] = SourceProfile(
            profile=profile_id,
            spark_line=spark_line,
            default_spark_version=default_spark_version,
            scala_binary=scala_binary,
            allowed_scala_binaries=allowed_scala_binaries,
        )
    if not profiles:
        raise ArtifactMetadataError("root pom.xml has no supported Spark shim profiles")
    return profiles


def select_source_profiles(
    pom_path: Path, spark_profile: str, scala_profile: str
) -> SourceProfile:
    profiles = source_profiles(pom_path)
    selected = profiles.get(spark_profile)
    if selected is None:
        raise ArtifactMetadataError(
            f"unsupported Spark shim profile {spark_profile}; source has "
            + ", ".join(sorted(profiles))
        )
    scala_match = re.fullmatch(r"scala-([0-9]+[.][0-9]+)", scala_profile)
    if scala_match is None:
        raise ArtifactMetadataError(
            "Scala profile must use the canonical scala-X.Y form"
        )
    source = _readable_regular_file(pom_path, "root pom.xml")
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as error:
        raise ArtifactMetadataError(
            f"could not parse root pom.xml: {source}"
        ) from error
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    profile_ids = {
        (node.findtext("m:id", default="", namespaces=namespace)).strip()
        for node in root.findall("m:profiles/m:profile", namespace)
    }
    if scala_profile not in profile_ids:
        raise ArtifactMetadataError(
            f"unsupported Scala profile {scala_profile} in current source"
        )
    requested_scala = scala_match.group(1)
    if requested_scala not in selected.allowed_scala_binaries:
        raise ArtifactMetadataError(
            f"{spark_profile} supports Scala "
            f"{', '.join(selected.allowed_scala_binaries)}, not {requested_scala}"
        )
    return SourceProfile(
        profile=selected.profile,
        spark_line=selected.spark_line,
        default_spark_version=selected.default_spark_version,
        scala_binary=requested_scala,
        allowed_scala_binaries=selected.allowed_scala_binaries,
    )


def validate_jar_source_support(info: BundleJarInfo, pom_path: Path) -> None:
    selected = source_profiles(pom_path).get(f"spark-{info.spark_line}")
    if selected is None:
        raise ArtifactMetadataError(
            f"bundle JAR Spark line {info.spark_line} has no source shim profile"
        )
    if info.scala_binary not in selected.allowed_scala_binaries:
        raise ArtifactMetadataError(
            f"bundle JAR Scala {info.scala_binary} is not supported by source profile "
            f"Spark {info.spark_line}; expected "
            f"{', '.join(selected.allowed_scala_binaries)}"
        )


def validate_native_jar_identity(native: NativeBuildInfo, jar: BundleJarInfo) -> None:
    if jar.backend_type != native.backend_type:
        raise ArtifactMetadataError("native and JVM backend_type do not match")
    if jar.gluten_revision != native.gluten_revision:
        raise ArtifactMetadataError("native and JVM Gluten revisions do not match")
    if jar.velox_revision != native.velox_revision:
        raise ArtifactMetadataError("native and JVM Velox revisions do not match")
