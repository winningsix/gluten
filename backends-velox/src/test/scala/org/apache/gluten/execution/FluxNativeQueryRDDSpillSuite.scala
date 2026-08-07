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
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable.ArrayBuffer

class FluxNativeQueryRDDSpillSuite extends AnyFunSuite {

  test("FLUX spill roots use Spark local directory namespace round robin") {
    val localRoot1 = Files.createTempDirectory("gluten-flux-spill-root-1")
    val localRoot2 = Files.createTempDirectory("gluten-flux-spill-root-2")
    try {
      val namespace = new Namespace(Array(localRoot1.toFile, localRoot2.toFile), "gluten-spill")

      val lease1 = FluxNativeQueryRDD.createSpillRootLease(namespace)
      val lease2 = FluxNativeQueryRDD.createSpillRootLease(namespace)
      val spillRoot1 = Paths.get(lease1.path)
      val spillRoot2 = Paths.get(lease2.path)

      assertUnderNamespace(spillRoot1, localRoot1)
      assertUnderNamespace(spillRoot2, localRoot2)
      assert(spillRoot1 != spillRoot2)
      assert(Files.isDirectory(spillRoot1))
      assert(Files.isDirectory(spillRoot2))
      lease1.close()
      lease2.close()
      assert(!Files.exists(spillRoot1))
      assert(!Files.exists(spillRoot2))
    } finally {
      FileUtils.deleteDirectory(localRoot1.toFile)
      FileUtils.deleteDirectory(localRoot2.toFile)
    }
  }

  test("FLUX spill-root lease cleans up when native creation fails before handoff") {
    val localRoot = Files.createTempDirectory("gluten-flux-spill-failure")
    try {
      val namespace = new Namespace(Array(localRoot.toFile), "gluten-spill")
      val lease = FluxNativeQueryRDD.createSpillRootLease(namespace)
      val spillRoot = Paths.get(lease.path)

      val failure = intercept[IllegalStateException] {
        lease.handoffAfterCreate[Long](_ => throw new IllegalStateException("create failed"))
      }

      assert(failure.getMessage == "create failed")
      assert(!Files.exists(spillRoot))
    } finally {
      FileUtils.deleteDirectory(localRoot.toFile)
    }
  }

  test("FLUX spill-root lease transfers ownership only after successful native creation") {
    val localRoot = Files.createTempDirectory("gluten-flux-spill-handoff")
    try {
      val namespace = new Namespace(Array(localRoot.toFile), "gluten-spill")
      val lease = FluxNativeQueryRDD.createSpillRootLease(namespace)
      val spillRoot = Paths.get(lease.path)

      val handle = lease.handoffAfterCreate {
        path =>
          assert(path == spillRoot.toFile.getAbsolutePath)
          42L
      }

      assert(handle == 42L)
      lease.close()
      assert(Files.isDirectory(spillRoot))
    } finally {
      FileUtils.deleteDirectory(localRoot.toFile)
    }
  }

  test("FLUX spill-root lease cleanup is idempotent") {
    val localRoot = Files.createTempDirectory("gluten-flux-spill-idempotent")
    val cleanupCalls = new AtomicInteger()
    try {
      val lease = new FluxSpillRootLease(localRoot.toFile, _ => cleanupCalls.incrementAndGet())

      lease.close()
      lease.close()

      assert(cleanupCalls.get() == 1)
    } finally {
      FileUtils.deleteDirectory(localRoot.toFile)
    }
  }

  test("FLUX setup cleanup preserves the primary failure and runs every action in order") {
    val primaryFailure = new IllegalStateException("listener installation failed")
    val failureReportError = new RuntimeException("failure report failed")
    val closeError = new RuntimeException("native close failed")
    val terminalReportError = new RuntimeException("terminal report failed")
    val cleanupOrder = ArrayBuffer.empty[String]

    val thrown = intercept[IllegalStateException] {
      FluxNativeQueryRDD.rethrowAfterCleanup(
        primaryFailure,
        () => {
          cleanupOrder += "failure-report"
          throw failureReportError
        },
        () => {
          cleanupOrder += "close"
          throw closeError
        },
        () => {
          cleanupOrder += "terminal-report"
          throw terminalReportError
        }
      )
    }

    assert(thrown eq primaryFailure)
    assert(cleanupOrder == Seq("failure-report", "close", "terminal-report"))
    assert(thrown.getSuppressed.toSeq == Seq(failureReportError, closeError, terminalReportError))
  }

  private def assertUnderNamespace(spillRoot: Path, localRoot: Path): Unit = {
    val expectedParent = localRoot.resolve("gluten-spill").toAbsolutePath.normalize()
    assert(spillRoot.toAbsolutePath.normalize().getParent == expectedParent)
    assert(spillRoot.getFileName.toString.startsWith("flux-"))
  }
}
