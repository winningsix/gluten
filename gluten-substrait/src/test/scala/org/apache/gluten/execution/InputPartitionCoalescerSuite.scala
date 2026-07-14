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

import org.apache.spark.sql.catalyst.plans.physical.{SinglePartition, UnknownPartitioning}
import org.apache.spark.sql.connector.read.InputPartition

import org.scalatest.funsuite.AnyFunSuite

class InputPartitionCoalescerSuite extends AnyFunSuite {

  test("coalesces 4522 adjacent partitions without losing source metadata") {
    val partitionCount = 4522
    val cost = 4L * 1024 * 1024
    val target = 128L * 1024 * 1024
    val input = (0 until partitionCount).map {
      index =>
        Seq(
          MockInputPartition(
            index,
            cost,
            fileSplit = s"file-$index",
            deleteSplit = Some(s"delete-$index"),
            partitionValue = s"partition-$index",
            locations = Array(s"host-${index % 4}")))
    }

    val result = coalesce(input, targetBytes = target)()
    val flattened = result.flatten.map(_.asInstanceOf[MockInputPartition])

    assert(result.size === 142)
    assert(flattened.map(_.id) === (0 until partitionCount))
    assert(flattened.map(_.fileSplit) === input.flatten.map(_.fileSplit))
    assert(flattened.map(_.deleteSplit) === input.flatten.map(_.deleteSplit))
    assert(flattened.map(_.partitionValue) === input.flatten.map(_.partitionValue))
    assert(flattened.zip(input.flatten).forall { case (actual, expected) => actual eq expected })

    val wrapped = result.zipWithIndex.map {
      case (partitions, index) => new SparkDataSourceRDDPartition(index, partitions)
    }
    assert(wrapped.flatMap(_.preferredLocations()) === input.flatten.flatMap(_.locations))
  }

  test("does not coalesce scans with partitioning or ordering semantics") {
    val input = mockGroups(4, cost = 10)

    assert(
      coalesce(input, eligibility = defaultEligibility.copy(hasOutputOrdering = true))() eq input)
    assert(
      coalesce(input, eligibility = defaultEligibility.copy(hasKeyGroupedPartitioning = true))() eq
        input)
    assert(
      coalesce(input, eligibility = defaultEligibility.copy(hasCommonPartitionValues = true))() eq
        input)
    assert(
      coalesce(input, eligibility = defaultEligibility.copy(requiresPartitionIdentity = true))() eq
        input)
    assert(
      coalesce(
        input,
        eligibility = defaultEligibility.copy(outputPartitioning = SinglePartition))() eq
        input)
  }

  test("only coalesces MPP scans outside single-task mode") {
    val input = mockGroups(4, cost = 10)

    assert(coalesce(input, eligibility = defaultEligibility.copy(mppEnabled = false))() eq input)
    assert(coalesce(input, eligibility = defaultEligibility.copy(singleTaskMode = true))() eq input)
  }

  test("does not combine adjacent partitions with different file formats") {
    val input: Seq[Seq[InputPartition]] = Vector(
      Vector(MockInputPartition(0, 10, fileFormat = "PARQUET")),
      Vector(MockInputPartition(1, 10, fileFormat = "PARQUET")),
      Vector(MockInputPartition(2, 10, fileFormat = "ORC")),
      Vector(MockInputPartition(3, 10, fileFormat = "ORC"))
    )

    val result = coalesce(input)()

    assert(result.map(_.size) === Seq(2, 2))
    assert(
      result.forall(
        group => group.map(_.asInstanceOf[MockInputPartition].fileFormat).distinct.size === 1))
  }

  test("keeps unknown exceptional and empty groups as barriers") {
    val knownBefore = MockInputPartition(0, 20)
    val unknown = MockInputPartition(1, 20)
    val exceptional = MockInputPartition(2, 20)
    val knownAfter1 = MockInputPartition(3, 20)
    val knownAfter2 = MockInputPartition(4, 20)
    val empty = Seq.empty[InputPartition]
    val input: Seq[Seq[InputPartition]] = Vector(
      Vector(knownBefore),
      Vector(unknown),
      Vector(exceptional),
      empty,
      Vector(knownAfter1),
      Vector(knownAfter2))

    val result = coalesce(input, targetBytes = 100) {
      case partition if partition eq unknown => None
      case partition if partition eq exceptional => throw new IllegalStateException("bad size")
      case partition: MockInputPartition => Some((partition.cost, partition.fileFormat))
      case _ => None
    }

    assert(
      result === Seq(
        Seq(knownBefore),
        Seq(unknown),
        Seq(exceptional),
        empty,
        Seq(knownAfter1, knownAfter2)))
  }

  test("does not overflow group or bin cost arithmetic") {
    val small = MockInputPartition(0, 1)
    val max = MockInputPartition(1, Long.MaxValue)
    val overflow = MockInputPartition(2, 1)
    val input: Seq[Seq[InputPartition]] =
      Vector(Vector(small), Vector(max, overflow), Vector(small.copy(id = 3)))

    assert(coalesce(input, targetBytes = Long.MaxValue)() eq input)

    val binOverflowInput: Seq[Seq[InputPartition]] =
      Vector(Vector(max), Vector(overflow))
    assert(coalesce(binOverflowInput, targetBytes = Long.MaxValue)() eq binOverflowInput)
  }

  private def mockGroups(count: Int, cost: Long): Seq[Seq[InputPartition]] = {
    (0 until count).map(index => Seq(MockInputPartition(index, cost)))
  }

  private val defaultEligibility = InputPartitionCoalescer.Eligibility(
    mppEnabled = true,
    singleTaskMode = false,
    outputPartitioning = UnknownPartitioning(4),
    hasOutputOrdering = false,
    hasKeyGroupedPartitioning = false,
    hasCommonPartitionValues = false,
    requiresPartitionIdentity = false
  )

  private def coalesce(
      input: Seq[Seq[InputPartition]],
      targetBytes: Long = 100,
      eligibility: InputPartitionCoalescer.Eligibility = defaultEligibility)(
      partitionInfo: InputPartition => Option[(Long, String)] = {
        case partition: MockInputPartition => Some((partition.cost, partition.fileFormat))
        case _ => None
      }): Seq[Seq[InputPartition]] = {
    InputPartitionCoalescer.coalesceAdjacentIfSupported(
      input,
      targetBytes,
      eligibility
    )(partitionInfo)
  }

  private case class MockInputPartition(
      id: Int,
      cost: Long,
      fileFormat: String = "PARQUET",
      fileSplit: String = "file",
      deleteSplit: Option[String] = None,
      partitionValue: String = "partition",
      locations: Array[String] = Array.empty[String])
    extends InputPartition {
    override def preferredLocations(): Array[String] = locations
  }
}
