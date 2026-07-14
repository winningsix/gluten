/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.gluten.execution

import org.apache.spark.util.Namespace

import org.apache.commons.io.FileUtils
import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.{Files, Path, Paths}

class MppNativeQueryRDDSpillSuite extends AnyFunSuite {

  test("MPP spill roots use Spark local directory namespace round robin") {
    val localRoot1 = Files.createTempDirectory("gluten-mpp-spill-root-1")
    val localRoot2 = Files.createTempDirectory("gluten-mpp-spill-root-2")
    try {
      val namespace = new Namespace(Array(localRoot1.toFile, localRoot2.toFile), "gluten-spill")

      val spillRoot1 = Paths.get(MppNativeQueryRDD.createSpillRoot(namespace))
      val spillRoot2 = Paths.get(MppNativeQueryRDD.createSpillRoot(namespace))

      assertUnderNamespace(spillRoot1, localRoot1)
      assertUnderNamespace(spillRoot2, localRoot2)
      assert(spillRoot1 != spillRoot2)
      assert(Files.isDirectory(spillRoot1))
      assert(Files.isDirectory(spillRoot2))
    } finally {
      FileUtils.deleteDirectory(localRoot1.toFile)
      FileUtils.deleteDirectory(localRoot2.toFile)
    }
  }

  private def assertUnderNamespace(spillRoot: Path, localRoot: Path): Unit = {
    val expectedParent = localRoot.resolve("gluten-spill").toAbsolutePath.normalize()
    assert(spillRoot.toAbsolutePath.normalize().getParent == expectedParent)
    assert(spillRoot.getFileName.toString.startsWith("mpp-"))
  }
}
