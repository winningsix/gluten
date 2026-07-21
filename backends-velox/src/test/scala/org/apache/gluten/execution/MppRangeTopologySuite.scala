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

import org.apache.gluten.extension.ExchangeSpec

import org.scalatest.funsuite.AnyFunSuite

class MppRangeTopologySuite extends AnyFunSuite {
  test("uses the effective RANGE partition count for native destinations") {
    val range = ExchangeSpec(
      id = 7,
      producerFragmentId = 2,
      consumerFragmentId = 1,
      exchangeType = "RANGE",
      numPartitions = 128,
      partitionKeys = Seq.empty,
      rangeBoundsJson = Some("{\"boundaries\":[]}"),
      rangeEffectivePartitions = Some(2)
    )
    val hash = ExchangeSpec(
      id = 8,
      producerFragmentId = 3,
      consumerFragmentId = 1,
      exchangeType = "HASH",
      numPartitions = 128,
      partitionKeys = Seq.empty)

    val Seq(effectiveRange, unchangedHash) =
      MppRangeTopology.applyEffectivePartitionCounts(Seq(range, hash))

    assert(effectiveRange.numPartitions === 2)
    assert(effectiveRange.rangeEffectivePartitions.contains(2))
    assert(effectiveRange.rangeBoundsJson === range.rangeBoundsJson)
    assert(unchangedHash === hash)
  }

  test("rejects missing or invalid effective RANGE partition counts") {
    def range(requested: Int, effective: Option[Int]): ExchangeSpec =
      ExchangeSpec(
        id = 9,
        producerFragmentId = 1,
        consumerFragmentId = 0,
        exchangeType = "RANGE",
        numPartitions = requested,
        partitionKeys = Seq.empty,
        rangeBoundsJson = Some("{\"boundaries\":[]}"),
        rangeEffectivePartitions = effective
      )

    intercept[IllegalStateException] {
      MppRangeTopology.applyEffectivePartitionCounts(Seq(range(4, None)))
    }
    intercept[IllegalArgumentException] {
      MppRangeTopology.applyEffectivePartitionCounts(Seq(range(4, Some(0))))
    }
    intercept[IllegalArgumentException] {
      MppRangeTopology.applyEffectivePartitionCounts(Seq(range(4, Some(5))))
    }
  }
}
