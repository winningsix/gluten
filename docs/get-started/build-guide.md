---
layout: page
title: Build Parameters for Velox Backend
nav_order: 4
parent: Getting-Started
---
## Build Parameters
### Native build parameters for buildbundle-veloxbe.sh or builddeps-veloxbe.sh
Please set them via `--`, e.g. `--build_type=Release`.

| Parameters             | Description                                                                                   | Default |
|------------------------|-----------------------------------------------------------------------------------------------|---------|
| build_type             | Build type for Velox & gluten cpp, CMAKE_BUILD_TYPE.                                          | Release |
| build_tests            | Build gluten cpp tests.                                                                       | OFF     |
| build_examples         | Build udf example.                                                                            | OFF     |
| build_benchmarks       | Build gluten cpp benchmarks.                                                                  | OFF     |
| enable_jemalloc_stats  | Print jemalloc stats for debugging.                                                           | OFF     |
| enable_qat             | Enable QAT for shuffle data de/compression.                                                   | OFF     |
| enable_s3              | Build with S3 support.                                                                        | OFF     |
| enable_gcs             | Build with GCS support.                                                                       | OFF     |
| enable_hdfs            | Build with HDFS support.                                                                      | OFF     |
| enable_abfs            | Build with ABFS support.                                                                      | OFF     |
| enable_vcpkg           | Enable vcpkg for static build.                                                                | OFF     |
| cudf_source            | Resolve cuDF from the Velox build (`BUNDLED`) or an installed package (`SYSTEM`).             | BUNDLED |
| cudf_version_info      | Data-only installed-cuDF source metadata required for checked `SYSTEM` builds.               | ""      |
| cudf_compatibility_check | Verify installed cuDF source alignment for `SYSTEM`; `OFF` fully skips the check.           | ON      |
| rebuild_if_mismatch    | Rebuild bundled cuDF after a recognized installed-artifact mismatch.                          | OFF     |
| run_setup_script       | Run setup script to install Velox dependencies.                                               | ON      |
| velox_repo             | Specify your own Velox repo to build.                                                         | ""      |
| velox_branch           | Specify your own Velox branch to build.                                                       | ""      |
| velox_home             | Specify your own Velox source path to build.                                                  | ""      |
| build_velox_tests      | Build Velox tests.                                                                            | OFF     |
| build_velox_benchmarks | Build Velox benchmarks (velox_tests and connectors will be disabled if ON)                    | OFF     |
| build_arrow            | Build arrow java/cpp and install the libs in local. Can turn it OFF after first build.        | ON      |
| spark_version          | Build for specified version of Spark(3.3, 3.4, 3.5, ALL). `ALL` means build for all versions. | ALL     |

### Velox build parameters for build-velox.sh
Please set them via `--`, e.g., `--velox_home=/YOUR/PATH`.

| Parameters       | Description                                                   | Default                                  |
|------------------|---------------------------------------------------------------|------------------------------------------|
| velox_home       | Specify Velox source path to build.                           | GLUTEN_SRC/ep/build-velox/build/velox_ep |
| build_type       | Velox build type, i.e., CMAKE_BUILD_TYPE.                     | Release                                  |
| enable_s3        | Build Velox with S3 support.                                  | OFF                                      |
| enable_gcs       | Build Velox with GCS support.                                 | OFF                                      |
| enable_hdfs      | Build Velox with HDFS support.                                | OFF                                      |
| enable_abfs      | Build Velox with ABFS support.                                | OFF                                      |
| cudf_source      | Resolve cuDF from `BUNDLED` source or a `SYSTEM` package.     | BUNDLED                                  |
| cudf_version_info | Installed-cuDF source metadata required for checked `SYSTEM` builds. | ""                              |
| cudf_compatibility_check | Verify installed cuDF source alignment; `OFF` fully skips the check. | ON                         |
| rebuild_if_mismatch | Rebuild bundled cuDF after a recognized SYSTEM mismatch.   | OFF                                      |
| run_setup_script | Run setup script to install Velox dependencies before build.  | ON                                       |
| build_test_utils | Build Velox with cmake arg -DVELOX_BUILD_TEST_UTILS=ON if ON. | OFF                                      |
| build_tests      | Build Velox test.                                             | OFF                                      |
| build_benchmarks | Build Velox benchmarks.                                       | OFF                                      |

### Build a complete bundle against an installed cuDF package

GPU builds use the cuDF revision selected by Velox by default. `SYSTEM` is an
opt-in mode for a host or container that already contains cuDF and UCXX CMake
packages plus their build dependencies.

The installed artifact must provide data-only cuDF source metadata at the
selected prefix, for example
`/usr/local/share/gluten/cudf-build-info`:

```properties
CUDF_COMMIT=<full-40-character-git-sha>
CUDF_VERSION=<version>
```

The build reads this file as data and never sources it as shell code.
`CUDF_VERSION` is diagnostic. Before compilation, the build derives the cuDF
commit selected by Velox and compares it with `CUDF_COMMIT`:

- The default (`--cudf_compatibility_check=ON` and
  `--rebuild_if_mismatch=OFF`) fails on missing, malformed, ambiguous, or
  mismatched source metadata.
- `--rebuild_if_mismatch=ON` changes only a recognized installed-artifact
  mismatch to one clean bundled-cuDF build. Failure to derive the selected
  Velox source identity remains fatal.
- `--cudf_compatibility_check=OFF` performs no source-metadata reads and uses
  the installed package with an explicit unverified warning. It cannot be
  combined with rebuild-on-mismatch.

Preparing and publishing a replacement dependency image remains an external
CI/CD concern.

Use the normal bundle entrypoint to build Velox, Gluten C++, and the Maven
bundle in one invocation:

```bash
export INSTALL_PREFIX=/opt/cudf-dependencies
export CMAKE_PREFIX_PATH="$INSTALL_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

./dev/buildbundle-veloxbe.sh \
  --enable_gpu=ON \
  --cudf_source=SYSTEM \
  --cudf_version_info="$INSTALL_PREFIX/share/gluten/cudf-build-info" \
  --spark_version=3.5
```

The native library is written under `cpp/build/releases/`, and the bundle JAR
is written under `package/target/`. Use `dev/builddeps-veloxbe.sh` with the same
options when only the native build is required.

Within one selected working tree, the entrypoints can reuse compatible native
state. They remove the selected Velox build directory before a SYSTEM/BUNDLED
transition or a mismatch-triggered bundled rebuild, so both native
configurations use the same effective source. This does not qualify arbitrary
cross-process incremental CMake reuse or a build tree copied from elsewhere.

### Build a local SYSTEM-cuDF dependency carrier

`dev/build-cudf-dependency-image.sh` builds a reusable local carrier containing
the toolchain, including CMake, Maven, JDK 17, and `patchelf`, and installed
dependencies needed by Spark-Gluten's SYSTEM-cuDF build path. Native
dependencies are under `/usr/local`, while the root-owned Maven repository
remains at `/root/.m2`. It uses
`dev/cudf-dependency-image/cudf_prebuilt.Dockerfile`, the existing
`dev/builddeps-veloxbe.sh` entrypoint, and the exact selected Velox source. The
recipe builds the cuDF and UCXX dependency targets, builds its checksum-pinned
UCX release with CUDA enabled, and builds UCXX against that same `/usr/local`
install. The final carrier intentionally exposes `libuct_cuda.so*` and
`libucm_cuda.so*` under `/usr/local/lib/ucx`.

The carrier also contains, unconditionally, the static AWS SDK C++ dependency
closure pinned by the selected Velox source. The current contract requires AWS
SDK C++ 1.11.654 for `s3`, `identity-management`, and `s3-crt`, including the
installed headers, CMake packages, archives, and their AWS CRT dependencies.
This carrier content does not enable S3 by itself: each consumer still selects
`--enable_s3=ON` or `OFF` when compiling Velox. There is no S3 producer profile
or automatic fallback.

The producer also replaces any partial Arrow installation inherited from the
base image with one coherent Arrow 15 build using Spark-Gluten's existing
`dev/build-arrow.sh` recipe and patches. Both `libarrow.a` and
`libarrow_bundled_dependencies.a`, their headers, and the patched Arrow Java
artifacts therefore come from the same prepared source tree. Before that build,
the producer clears any inherited `org/apache/arrow` Maven state. It retains the
resulting `/root/.m2` in the carrier and requires exact nonempty JAR and POM files
for these `15.0.0-gluten` artifacts:

- `arrow-memory-unsafe`
- `arrow-memory-core`
- `arrow-vector`
- `arrow-c-data`
- `arrow-dataset`

Each JAR must be readable by `jar tf`, and the Arrow Maven subtree must contain
no `.lastUpdated` files. `/root/.m2` is intentionally the only prepared Maven
repository interface. The carrier does not provide a global Maven seed,
repository-population helper, or arbitrary-user Maven path.

Tests, examples, benchmarks, compiled Velox outputs, and compiled Gluten outputs
are not carrier contents. The patched Arrow dependency JARs above are carrier
inputs for the later Gluten Maven build, not compiled Gluten outputs. The
producer writes the same
`/usr/local/share/gluten/cudf-build-info` source metadata consumed above; AWS
and Arrow dependencies do not change that cuDF-only marker or introduce a
second marker or format.

Provide the full cuDF commit declared by the selected Velox source, its
diagnostic version, and explicit CUDA architectures:

Set `CUDA_ARCH_LIST` from the deployment targets that will consume the carrier,
not from the machine running the producer. Multiple targets can be supplied as
a comma-separated list, for example `80-real,90-real`.

```bash
: "${CUDA_ARCH_LIST:?Set CUDA_ARCH_LIST to the target GPU architectures}"

./dev/build-cudf-dependency-image.sh \
  --velox_dir=/path/to/velox \
  --image=gluten-cudf-dependencies:local \
  --cudf_commit=<full-40-character-git-sha> \
  --cudf_version=26.08 \
  --cuda_arch="${CUDA_ARCH_LIST}" \
  --num_threads=12
```

Trusted callers that replace Maven repositories can provide one standard
Maven `settings.xml` file. For example, a caller-owned file can select a
repository manager without changing the carrier recipe:

```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <mirrors>
    <mirror>
      <id>caller-repository</id>
      <mirrorOf>*</mirrorOf>
      <url>https://maven.example.invalid/repository/proxy</url>
    </mirror>
  </mirrors>
</settings>
```

Pass its path explicitly to both the producer and protected smoke:

```bash
./dev/build-cudf-dependency-image.sh \
  --velox_dir=/path/to/velox \
  --image=gluten-cudf-dependencies:local \
  --cudf_commit=<full-40-character-git-sha> \
  --cudf_version=26.08 \
  --cuda_arch="${CUDA_ARCH_LIST}" \
  --maven_settings=/secure/path/settings.xml
```

The producer validates a readable, nonempty regular file before Docker starts
and forwards it as a BuildKit secret only to the Arrow Java build. The file is
temporarily installed as root's standard Maven settings and removed in the
same build step; it is not copied into the carrier. If the option is absent,
Maven's existing default resolution behavior is unchanged. This interface does
not mount a host Maven repository or define an environment-variable alias.

The producer validates the arguments and installed-cuDF source metadata before
building. It rejects a selected Velox tree containing caller-local `_build`
state. GPU-independent image assembly checks representative cuDF and UCXX
headers; the installed cuDF, UCXX, and NVTX3 CMake packages and shared
libraries; the coherent static Arrow C++ archives and representative headers;
the five exact Arrow Java JAR/POM pairs and their readability; CUDA UCX build
configuration; both CUDA modules at the exact module path; and the absence of
compiled Velox or Gluten artifacts. For AWS, it keeps small static S3/S3-CRT
archive canaries, then configures, compiles, and links a temporary consumer
containing `S3Client`, `S3CrtClient`, and the assume-role credentials provider
used by Velox. This semantic probe admits the installed headers, CMake
components, service archives, and transitive AWS CRT link closure without
maintaining a list of every CRT archive. The probe is not run and requires no
AWS credentials or network access.

After assembly, the host wrapper runs the installed checker as root with
`HOME=/root` and `--gpus all`. This is the supported consumer identity for the
prepared Maven repository. The checker requires `ucx_info -d` to report both
`cuda_copy` and `cuda_ipc`. A working NVIDIA container runtime and GPU are
therefore required to complete producer validation; a correct CUDA UCX build
cannot advertise those live transports to a container that has no driver
injection. The carrier does not copy CUDA driver libraries or stubs; NVIDIA
Container Runtime supplies the host-compatible driver libraries when the
container runs.

The command creates and checks only a local Docker tag. Registry credentials,
publication, promotion, and registration remain outside these upstream
scripts.

The default toolchain base is the Linux/amd64 Velox adapters image used by the
existing cuDF recipe. That base supplies glibc and the ELF dynamic loader; they
are not a portable runtime-bundle contract. The producer installs and selects
JDK 17 in the carrier. NVIDIA driver libraries are not carrier content and are
injected by the host's NVIDIA container runtime. A `--base_image` override must
provide a compatible CentOS toolchain with `dnf`, `uv`, CUDA, and GCC toolset
14, and it requires the same fresh SYSTEM smoke below.

Run the single fresh SYSTEM-cuDF buildbundle smoke separately in protected or
nightly validation. Premerge validation should run the source/fixture checks in
`dev/test-system-cudf.sh` and `dev/cudf-dependency-image/test.sh`; it should not
run this full consumer build.

The producer host does not need to determine the carrier architecture list.
However, the GPU that runs this smoke must be represented in
`CUDA_ARCH_LIST`; otherwise, run the smoke on a represented deployment GPU.
Both the Spark-Gluten and selected Velox directories must be clean Git
worktrees. Before starting the container, the smoke logs their exact full HEAD
SHAs, resolves the requested carrier tag to its full Docker image ID, logs that
mapping, and runs the immutable image ID. The mounted source inputs are archives
of those exact commits, so ignored caller-local files cannot alter the build.
These are plain log lines, not a new receipt or identity schema.

```bash
./dev/cudf-dependency-image/smoke-system.sh \
  --image=gluten-cudf-dependencies:local \
  --velox_dir=/path/to/velox \
  --cuda_arch="${CUDA_ARCH_LIST}"
```

The smoke uses all CPUs reported by `nproc` by default. Use
`--num_threads=N` only to impose an explicit positive build-parallelism limit.

To use the same caller-owned repository replacement in the smoke, add
`--maven_settings=/secure/path/settings.xml`. The resolved file alone is
bind-mounted read-only at `/root/.m2/settings.xml`; its contents are not logged.
Omitting the option retains the canonical smoke command above.

HDFS build support is consumer-configurable. Omitting `--enable_hdfs` keeps the
canonical smoke in its default `OFF` mode. Protected validation also runs the
same command against the same dependency carrier with `--enable_hdfs=ON`. The
`ON` mode qualifies the native and Maven build configuration only. The
dependency carrier does not provide Hadoop, `libhdfs.so`, a Hadoop classpath,
or HDFS configuration; downstream runtime images supply those components and
own live HDFS I/O qualification.

The smoke starts a fresh root container with `HOME=/root`, selects JDK 17, and
builds Spark 3.5.5 for Scala 2.12 with Maven profiles
`backends-velox,spark-3.5,java-17`. It invokes
`dev/buildbundle-veloxbe.sh` with `--run_setup_script=OFF`,
`--build_arrow=OFF`, `--enable_s3=ON`, the selected
`--enable_hdfs=ON|OFF` value,
`--cudf_source=SYSTEM`, `--cudf_compatibility_check=ON`, and
`--rebuild_if_mismatch=OFF`. Tests, examples, and benchmarks remain disabled.
The only requested native targets are `velox` and `velox_cudf_exec`; the latter
brings in the complete cuDF static-library set imported by Gluten. The smoke
does not enable Velox testing or build `velox_cudf_null_mask_test` or
`velox_s3config_test`.

The smoke verifies that both Velox and Gluten retained SYSTEM cuDF with S3
compile/link support enabled and that both CMake caches retained the selected
HDFS build mode; both Arrow static archives resolve coherently from
`/usr/local/lib*`; and the AWS SDK CMake package resolves from `/usr/local`. It
requires the real `cpp/build/releases/libgluten.so` with no unresolved `ldd`
dependencies in this carrier/build environment, a readable Gluten Arrow module
JAR from the successful Maven reactor, and exactly one readable
`gluten-velox-bundle-spark3.5_2.12-*.jar`. HDFS `ON` is build-qualified, not a
live HDFS read/write or downstream-runtime qualification, and the in-carrier
`ldd` result is not portable clean-runtime closure proof. The smoke performs no
live S3 or HDFS request, Spark/JNI runtime test, or paid service run and requires
no AWS credentials or endpoint.

A matching `CUDF_COMMIT` proves source alignment only. It does not prove
compiler, CUDA, SM, flags, patches, ABI, or binary equivalence. The local image
and smoke likewise do not qualify arbitrary cross-process incremental CMake
reuse.

### Maven build parameters
The below parameters can be set via `-P` for mvn.

| Parameters          | Description                           | Default state |
|---------------------|---------------------------------------|---------------|
| backends-velox      | Build Gluten Velox backend.           | disabled      |
| backends-clickhouse | Build Gluten ClickHouse backend.      | disabled      |
| celeborn            | Build Gluten with Celeborn.           | disabled      |
| uniffle             | Build Gluten with Uniffle.            | disabled      |
| delta               | Build Gluten with Delta Lake support. | disabled      |
| iceberg             | Build Gluten with Iceberg support.    | disabled      |
| hudi                | Build Gluten with Hudi support.       | disabled      |
| spark-3.3           | Build Gluten for Spark 3.3.           | disabled       |
| spark-3.4           | Build Gluten for Spark 3.4.           | disabled      |
| spark-3.5           | Build Gluten for Spark 3.5.           | enabled      |

## Gluten Jar for Deployment
The gluten jar built out is under `GLUTEN_SRC/package/target/`.
It's name pattern is `gluten-<backend_type>-bundle-spark<spark.bundle.version>_<scala.binary.version>-<os.detected.release>_<os.detected.release.version>-<project.version>.jar`.

| Spark Version | spark.bundle.version | scala.binary.version |
|---------------|----------------------|----------------------|
| 3.3.1         | 3.3                  | 2.12                 |
| 3.4.4         | 3.4                  | 2.12                 |
| 3.5.5         | 3.5                  | 2.12                 |
