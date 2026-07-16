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

import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.mockito.Mockito.mock
import org.scalatest.funsuite.AnyFunSuite

import java.util.concurrent.atomic.AtomicInteger

class MppAlignedInputRDDSuite extends AnyFunSuite {

  test("aligned input creates parent iterators lazily") {
    val computeCalls = new AtomicInteger()
    val parent =
      new CountingColumnarRDD(mock(classOf[SparkContext]), numPartitions = 500, computeCalls)
    val aligned = new MppAlignedInputRDD(parent, Array((0 until 500).toArray))

    val output = aligned.compute(aligned.partitions.head, mock(classOf[TaskContext]))
    assert(computeCalls.get() == 0)

    assert(output.hasNext)
    assert(computeCalls.get() == 1)
    output.next()
    assert(computeCalls.get() == 1)

    assert(output.hasNext)
    assert(computeCalls.get() == 2)
  }

  private case class CountingPartition(override val index: Int) extends Partition

  private class CountingColumnarRDD(
      sparkContext: SparkContext,
      numPartitions: Int,
      computeCalls: AtomicInteger)
    extends RDD[ColumnarBatch](sparkContext, Nil) {

    override protected def getPartitions: Array[Partition] =
      Array.tabulate(numPartitions)(CountingPartition)

    override def compute(
        split: Partition,
        context: org.apache.spark.TaskContext): Iterator[ColumnarBatch] = {
      computeCalls.incrementAndGet()
      Iterator.single(null.asInstanceOf[ColumnarBatch])
    }
  }
}
