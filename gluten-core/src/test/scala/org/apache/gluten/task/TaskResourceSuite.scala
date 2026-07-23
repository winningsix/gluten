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
package org.apache.gluten.task

import org.apache.spark.memory.{MemoryConsumer, MemoryMode}
import org.apache.spark.sql.catalyst.plans.SQLHelper
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.task.{TaskResource, TaskResources}
import org.apache.spark.util.SparkTaskUtil

import org.scalatest.funsuite.AnyFunSuite

import java.util.Properties
import java.util.UUID
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, Executors}
import java.util.concurrent.atomic.AtomicReference

import scala.collection.JavaConverters._

class TaskResourceSuite extends AnyFunSuite with SQLHelper {
  test("Run unsafe") {
    val out = TaskResources.runUnsafe {
      1
    }
    assert(out == 1)
  }

  test("Run unsafe - task context") {
    TaskResources.runUnsafe {
      assert(TaskResources.inSparkTask())
      assert(TaskResources.getLocalTaskContext() != null)
    }
  }

  test("Synthetic task contexts have unique negative attempt IDs") {
    val threadCount = 16
    val start = new CountDownLatch(1)
    val done = new CountDownLatch(threadCount)
    val executor = Executors.newFixedThreadPool(threadCount)
    val taskAttemptIds = new ConcurrentLinkedQueue[Long]()
    val failure = new AtomicReference[Throwable]()
    val taskAttemptIdField = classOf[org.apache.spark.memory.TaskMemoryManager]
      .getDeclaredField("taskAttemptId")
    taskAttemptIdField.setAccessible(true)
    try {
      (0 until threadCount).foreach {
        _ =>
          executor.execute(
            () => {
              try {
                start.await()
                val context = SparkTaskUtil.createTestTaskContext(new Properties())
                taskAttemptIds.add(context.taskAttemptId())
                assert(
                  taskAttemptIdField.getLong(SparkTaskUtil.getTaskMemoryManager(context)) ==
                    context.taskAttemptId())
              } catch {
                case t: Throwable => failure.compareAndSet(null, t)
              } finally {
                done.countDown()
              }
            })
      }
      start.countDown()
      done.await()
      if (failure.get() != null) {
        throw failure.get()
      }
      val ids = taskAttemptIds.asScala.toSeq
      assert(ids.size == threadCount)
      assert(ids.forall(_ < 0))
      assert(ids.distinct.size == ids.size)
    } finally {
      executor.shutdownNow()
    }
  }

  test("Run unsafe - propagate Spark config") {
    val total = 128 * 1024 * 1024
    withSQLConf(
      "spark.memory.offHeap.enabled" -> "true",
      "spark.memory.offHeap.size" -> s"$total",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      TaskResources.runUnsafe {
        assert(TaskResources.inSparkTask())
        assert(TaskResources.getLocalTaskContext() != null)

        val tmm = SparkTaskUtil.getTaskMemoryManager(TaskResources.getLocalTaskContext())
        val consumer = new MemoryConsumer(tmm, MemoryMode.OFF_HEAP) {
          override def spill(size: Long, trigger: MemoryConsumer): Long = 0L
        }
        assert(consumer.acquireMemory(total) == total)
        assert(consumer.acquireMemory(1) == 0)

        assert(!SQLConf.get.adaptiveExecutionEnabled)
      }
    }
  }

  test("Run unsafe - register resource") {
    var unregisteredCount = 0
    TaskResources.runUnsafe {
      TaskResources.addResource(
        UUID.randomUUID().toString,
        new TaskResource {
          override def release(): Unit = unregisteredCount += 1

          override def resourceName(): String = "test resource 1"
        })
      TaskResources.addResource(
        UUID.randomUUID().toString,
        new TaskResource {
          override def release(): Unit = unregisteredCount += 1

          override def resourceName(): String = "test resource 2"
        })
    }
    assert(unregisteredCount == 2)
  }

  test("Bind an active task context to a callback thread") {
    var released = 0
    TaskResources.runUnsafe {
      val owner = TaskResources.getLocalTaskContext()
      val failure = new AtomicReference[Throwable]()
      val callback = new Thread(
        () => {
          try {
            assert(!TaskResources.inSparkTask())
            TaskResources.runWithTaskContext(owner) {
              assert(TaskResources.getLocalTaskContext() eq owner)
              TaskResources.addResource(
                "callback-resource",
                new TaskResource {
                  override def release(): Unit = released += 1

                  override def resourceName(): String = "callback resource"
                })
            }
            assert(!TaskResources.inSparkTask())
          } catch {
            case t: Throwable => failure.set(t)
          }
        },
        "task-context-callback-test"
      )
      callback.start()
      callback.join()
      if (failure.get() != null) {
        throw failure.get()
      }
      assert(released == 0)
    }
    assert(released == 1)
  }
}
