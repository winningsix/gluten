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
import java.util.concurrent.atomic.AtomicReference

class MppNativeQueryRDDTaskContextSuite extends AnyFunSuite with SQLHelper {

  test("MPP input bridge binds the owning Spark task context on native callback threads") {
    var released = 0
    TaskResources.runUnsafe {
      val owner = TaskResources.getLocalTaskContext()
      val nextSentinel = new IllegalStateException("delegated next sentinel")
      val delegated = new JIterator[ColumnarBatch] {
        override def hasNext: Boolean = {
          assert(TaskResources.getLocalTaskContext() eq owner)
          TaskResources.addResource(
            "mpp-callback-resource",
            new TaskResource {
              override def release(): Unit = released += 1

              override def resourceName(): String = "MPP callback resource"
            })
          false
        }

        override def next(): ColumnarBatch = {
          assert(TaskResources.getLocalTaskContext() eq owner)
          throw nextSentinel
        }
      }
      val bridge =
        MppNativeQueryRDD.createTaskContextAwareInputIterator("velox", delegated, owner)
      val failure = new AtomicReference[Throwable]()
      val callback = new Thread(
        () => {
          try {
            assert(!TaskResources.inSparkTask())
            assert(!bridge.hasNext())
            val observed = intercept[IllegalStateException](bridge.next())
            assert(observed eq nextSentinel)
            assert(!TaskResources.inSparkTask())
          } catch {
            case t: Throwable => failure.set(t)
          }
        },
        "mpp-native-callback-test"
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
