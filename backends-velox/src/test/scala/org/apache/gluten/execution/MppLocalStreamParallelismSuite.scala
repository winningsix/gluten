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

import org.apache.gluten.extension.NativeFragment

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types.LongType

import org.scalatest.funsuite.AnyFunSuite

class MppLocalStreamParallelismSuite extends AnyFunSuite {

  test("JVM local stream restricts only its consumer fragment") {
    val key = AttributeReference("key", LongType, nullable = false)()
    val source = MppExchangeSourceTransformer(0, Seq(key))
    val exec = MppNativeQueryExec(source, Seq.empty, Seq.empty)
    val fragments = Seq(
      NativeFragment(0, source, Seq(key), parallelism = 2),
      NativeFragment(1, source, Seq(key), parallelism = 4),
      NativeFragment(2, source, Seq(key), parallelism = 1))

    val adjusted = exec.restrictParallelismForLocalStreamConsumers(fragments, Set(1))

    assert(adjusted.map(_.parallelism) == Seq(2, 1, 1))
  }

  test("no JVM local stream preserves every fragment parallelism") {
    val key = AttributeReference("key", LongType, nullable = false)()
    val source = MppExchangeSourceTransformer(0, Seq(key))
    val exec = MppNativeQueryExec(source, Seq.empty, Seq.empty)
    val fragments = Seq(
      NativeFragment(0, source, Seq(key), parallelism = 2),
      NativeFragment(1, source, Seq(key), parallelism = 4))

    val adjusted = exec.restrictParallelismForLocalStreamConsumers(fragments, Set.empty)

    assert(adjusted.map(_.parallelism) == Seq(2, 4))
  }
}
