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

"""Classify installed cuDF compatibility with the selected Velox source."""

import argparse
from pathlib import Path
import re
import sys

FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
COMMIT_SET = re.compile(
    r"^[ \t]*set[ \t]*\([ \t]*VELOX_cudf_COMMIT\b(?P<body>[^)\r\n]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
COMPATIBLE = 0
OBSERVED_INCOMPATIBLE = 10
EXPECTED_IDENTITY_INVALID = 11


def read_expected_commit(velox_home):
    dependency_file = velox_home / "CMake" / "resolve_dependency_modules" / "cudf.cmake"
    try:
        text = dependency_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return None, "<missing>", str(error)

    values = []
    for match in COMMIT_SET.finditer(text):
        body = match.group("body").strip()
        tokens = body.split()
        value = tokens[0].strip('"') if len(tokens) == 1 else "<malformed>"
        values.append(value)

    if not values:
        return None, "<missing>", "VELOX_cudf_COMMIT is missing"
    if len(values) != 1:
        return (
            None,
            f"<ambiguous:{len(values)}>",
            f"found {len(values)} VELOX_cudf_COMMIT declarations",
        )
    if not FULL_SHA.fullmatch(values[0]):
        return None, values[0], "VELOX_cudf_COMMIT is not a full SHA"
    return values[0].lower(), values[0].lower(), None


def read_observed_commit(version_info):
    if not version_info:
        return None, "<missing>", None, "--cudf_version_info is required"

    marker = Path(version_info).expanduser().resolve()
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return None, "<missing>", None, str(error)

    commits = []
    version = None
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key.strip() == "CUDF_COMMIT":
            commits.append(value.strip())
        elif key.strip() == "CUDF_VERSION":
            version = value.strip()

    if not commits:
        return None, "<missing>", version, "CUDF_COMMIT is missing"
    if len(commits) != 1:
        return (
            None,
            f"<ambiguous:{len(commits)}>",
            version,
            f"found {len(commits)} CUDF_COMMIT entries",
        )
    if not FULL_SHA.fullmatch(commits[0]):
        return None, commits[0], version, "CUDF_COMMIT is not a full SHA"
    return commits[0].lower(), commits[0].lower(), version, None


def verify(velox_home, version_info):
    expected, expected_display, expected_error = read_expected_commit(velox_home)
    identity = f"expected={expected_display}; observed=<unchecked>"
    if expected_error:
        print(
            "ERROR: SYSTEM cuDF compatibility check failed: "
            f"{expected_error}; {identity}; action=fail",
            file=sys.stderr,
        )
        return EXPECTED_IDENTITY_INVALID

    observed, observed_display, version, observed_error = read_observed_commit(
        version_info
    )
    identity = f"expected={expected_display}; observed={observed_display}"
    if observed_error:
        print(
            "ERROR: SYSTEM cuDF compatibility check failed: "
            f"{observed_error}; {identity}; action=reject-system-artifact",
            file=sys.stderr,
        )
        return OBSERVED_INCOMPATIBLE
    if expected != observed:
        print(
            "ERROR: SYSTEM cuDF compatibility check failed: "
            "installed cuDF does not match the selected Velox source; "
            f"{identity}; action=reject-system-artifact",
            file=sys.stderr,
        )
        return OBSERVED_INCOMPATIBLE

    print(
        "SYSTEM cuDF compatibility verified: "
        f"{identity}; version={version or '<unspecified>'}; action=use-system"
    )
    return COMPATIBLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velox-home", type=Path)
    parser.add_argument("--version-info", default="")
    parser.add_argument(
        "--skip",
        action="store_true",
        help="skip all installed-cuDF identity checks",
    )
    args = parser.parse_args()
    if args.skip:
        print(
            "WARNING: SYSTEM cuDF compatibility check skipped; "
            "installed artifact identity is unverified; action=use-system-unverified",
            file=sys.stderr,
        )
        return 0
    if not args.velox_home:
        parser.error("--velox-home is required unless --skip is used")
    return verify(args.velox_home, args.version_info)


if __name__ == "__main__":
    raise SystemExit(main())
