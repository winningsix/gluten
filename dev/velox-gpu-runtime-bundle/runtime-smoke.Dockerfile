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

# Disposable environment for validating a mounted full bundle outside its
# build prefix. The outer smoke chooses one exact qualified Spark/Scala/JDK 17
# image from embedded artifact metadata. This image contains the ELF/RTCX/Spark
# smoke scripts, but intentionally contains no build inputs, sources, CUDA
# toolkit, or bundle.
ARG SPARK_RUNTIME_IMAGE
FROM ${SPARK_RUNTIME_IMAGE}

USER 0
ENV PATH="${SPARK_HOME}/bin:${PATH}"
RUN command -v python3 >/dev/null \
    && command -v spark-submit >/dev/null \
    && command -v jar >/dev/null \
    && command -v ldd >/dev/null \
    && java -version 2>&1 | grep -E 'version "17([.]|")' >/dev/null

COPY artifact_metadata.py runtime-smoke.py runtime-query.py /opt/gluten-runtime-smoke/

ENTRYPOINT ["python3", "/opt/gluten-runtime-smoke/runtime-smoke.py"]
CMD ["--container-runtime"]
