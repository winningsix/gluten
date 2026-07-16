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

import org.apache.spark.sql.catalyst.plans.SQLHelper
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.task.{TaskResource, TaskResources}

import org.scalatest.funsuite.AnyFunSuite

import java.util.{Iterator => JIterator}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicReference}

class MppNativeQueryRDDTaskContextSuite extends AnyFunSuite with SQLHelper {

  test("MPP close holds native memory before destroying the coordinator") {
    var events = Vector.empty[String]

    MppNativeQueryRDD.closeAfterHoldingMemory(
      () => events :+= "hold",
      () => events :+= "close",
      () => events :+= "mark-closed")

    assert(events == Seq("hold", "close", "mark-closed"))

    val closeFailure = new IllegalStateException("native close failed")
    events = Vector.empty
    val thrown = intercept[IllegalStateException] {
      MppNativeQueryRDD.closeAfterHoldingMemory(
        () => events :+= "hold",
        () => {
          events :+= "close"
          throw closeFailure
        },
        () => events :+= "mark-closed")
    }
    assert(thrown eq closeFailure)
    assert(events == Seq("hold", "close", "mark-closed"))

    val holdFailure = new IllegalStateException("memory hold failed")
    events = Vector.empty
    val holdThrown = intercept[IllegalStateException] {
      MppNativeQueryRDD.closeAfterHoldingMemory(
        () => {
          events :+= "hold"
          throw holdFailure
        },
        () => events :+= "close",
        () => events :+= "mark-closed")
    }
    assert(holdThrown eq holdFailure)
    assert(events == Seq("hold"))
  }

  test("MPP input bridge binds the owning Spark task context on native callback threads") {
    var released = 0
    val ownerThread = Thread.currentThread()
    val originalOwnerContextClassLoader = ownerThread.getContextClassLoader
    val ownerContextClassLoader = new ClassLoader(originalOwnerContextClassLoader) {}
    ownerThread.setContextClassLoader(ownerContextClassLoader)
    try {
      TaskResources.runUnsafe {
        val owner = TaskResources.getLocalTaskContext()
        val nextSentinel = new IllegalStateException("delegated next sentinel")
        val visibleClassName = classOf[MppNativeQueryRDDTaskContextSuite].getName
        val delegated = new JIterator[ColumnarBatch] {
          private def assertOwnerThreadContext(): Unit = {
            assert(TaskResources.getLocalTaskContext() eq owner)
            assert(Thread.currentThread().getContextClassLoader eq ownerContextClassLoader)
            // scalastyle:off classforname
            val visibleClass =
              Class.forName(visibleClassName, true, Thread.currentThread().getContextClassLoader)
            // scalastyle:on classforname
            assert(visibleClass eq classOf[MppNativeQueryRDDTaskContextSuite])
          }

          override def hasNext: Boolean = {
            assertOwnerThreadContext()
            TaskResources.addResource(
              "mpp-callback-resource",
              new TaskResource {
                override def release(): Unit = released += 1

                override def resourceName(): String = "MPP callback resource"
              })
            false
          }

          override def next(): ColumnarBatch = {
            assertOwnerThreadContext()
            throw nextSentinel
          }
        }
        val bridge =
          MppNativeQueryRDD.createTaskContextAwareInputIterator("velox", delegated, owner)
        val failure = new AtomicReference[Throwable]()
        val hostileContextClassLoader = new ClassLoader(null) {}
        // scalastyle:off classforname
        assertThrows[ClassNotFoundException] {
          Class.forName(visibleClassName, true, hostileContextClassLoader)
        }
        // scalastyle:on classforname
        val callback = new Thread(
          () => {
            try {
              assert(!TaskResources.inSparkTask())
              assert(Thread.currentThread().getContextClassLoader eq hostileContextClassLoader)
              assert(!bridge.hasNext())
              assert(!TaskResources.inSparkTask())
              assert(Thread.currentThread().getContextClassLoader eq hostileContextClassLoader)
              val observed = intercept[IllegalStateException](bridge.next())
              assert(observed eq nextSentinel)
              assert(!TaskResources.inSparkTask())
              assert(Thread.currentThread().getContextClassLoader eq hostileContextClassLoader)
            } catch {
              case t: Throwable => failure.set(t)
            }
          },
          "mpp-native-callback-test"
        )
        callback.setContextClassLoader(hostileContextClassLoader)
        callback.start()
        callback.join()
        if (failure.get() != null) {
          throw failure.get()
        }
        assert(released == 0)
      }
    } finally {
      ownerThread.setContextClassLoader(originalOwnerContextClassLoader)
    }
    assert(released == 1)
  }

  test("MPP input bridge rejects a null owner thread context classloader") {
    val ownerThread = Thread.currentThread()
    val originalOwnerContextClassLoader = ownerThread.getContextClassLoader
    val delegatedUsed = new AtomicBoolean(false)
    val delegated = new JIterator[ColumnarBatch] {
      override def hasNext: Boolean = {
        delegatedUsed.set(true)
        false
      }

      override def next(): ColumnarBatch = {
        delegatedUsed.set(true)
        throw new IllegalStateException("delegated iterator must not be used")
      }
    }
    TaskResources.runUnsafe {
      val owner = TaskResources.getLocalTaskContext()
      ownerThread.setContextClassLoader(null)
      try {
        val error = intercept[IllegalArgumentException] {
          MppNativeQueryRDD.createTaskContextAwareInputIterator("velox", delegated, owner)
        }
        assert(
          error.getMessage.contains(
            "MPP input callback bridge requires a non-null owner thread context classloader"))
        assert(!delegatedUsed.get())
      } finally {
        ownerThread.setContextClassLoader(originalOwnerContextClassLoader)
      }
    }
    assert(ownerThread.getContextClassLoader eq originalOwnerContextClassLoader)
  }

  test("MPP input bridge binds the owner classloader when the task context is already bound") {
    val ownerThread = Thread.currentThread()
    val originalOwnerContextClassLoader = ownerThread.getContextClassLoader
    val ownerContextClassLoader = new ClassLoader(originalOwnerContextClassLoader) {}
    val hostileContextClassLoader = new ClassLoader(null) {}
    ownerThread.setContextClassLoader(ownerContextClassLoader)
    try {
      TaskResources.runUnsafe {
        val owner = TaskResources.getLocalTaskContext()
        val delegated = new JIterator[ColumnarBatch] {
          override def hasNext: Boolean = {
            assert(TaskResources.getLocalTaskContext() eq owner)
            assert(Thread.currentThread().getContextClassLoader eq ownerContextClassLoader)
            false
          }

          override def next(): ColumnarBatch = throw new NoSuchElementException
        }
        val bridge =
          MppNativeQueryRDD.createTaskContextAwareInputIterator("velox", delegated, owner)
        ownerThread.setContextClassLoader(hostileContextClassLoader)
        try {
          assert(TaskResources.getLocalTaskContext() eq owner)
          assert(!bridge.hasNext())
          assert(TaskResources.getLocalTaskContext() eq owner)
          assert(ownerThread.getContextClassLoader eq hostileContextClassLoader)
        } finally {
          ownerThread.setContextClassLoader(ownerContextClassLoader)
        }
      }
    } finally {
      ownerThread.setContextClassLoader(originalOwnerContextClassLoader)
    }
    assert(ownerThread.getContextClassLoader eq originalOwnerContextClassLoader)
  }
}
