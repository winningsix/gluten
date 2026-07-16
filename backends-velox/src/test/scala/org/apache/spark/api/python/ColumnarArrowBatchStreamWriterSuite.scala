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
package org.apache.spark.api.python

import org.apache.gluten.memory.arrow.alloc.ArrowBufferAllocators
import org.apache.gluten.vectorized.ArrowWritableColumnVector

import org.apache.spark.sql.types.{LongType, StructType}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.task.TaskResources

import org.apache.arrow.vector.BigIntVector
import org.apache.arrow.vector.ipc.ArrowStreamReader
import org.scalatest.funsuite.AnyFunSuite

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataOutputStream}

import scala.collection.mutable.ArrayBuffer

class ColumnarArrowBatchStreamWriterSuite extends AnyFunSuite {
  private val schema = new StructType().add("value", LongType, nullable = false)

  test("Spark 4 writer serializes one batch per call and writes EOS once") {
    TaskResources.runUnsafe {
      val values = Seq(11L, 22L, 33L)
      val acknowledgements = ArrayBuffer.empty[Long]
      val batches = newBatches(values, acknowledgements)
      val bytes = new ByteArrayOutputStream()
      val dataOut = new DataOutputStream(bytes)
      val writer = new ColumnarArrowBatchStreamWriter(schema, "UTC")

      try {
        val input = batches.iterator
        values.indices.foreach {
          index =>
            assert(writer.writeNext(input, dataOut))
            assert(acknowledgements == values.take(index + 1))
        }
        assert(!writer.writeNext(input, dataOut))
        assert(acknowledgements == values)
        assert(readValues(bytes.toByteArray) == values)
      } finally {
        writer.close()
        batches.foreach(_.close())
        dataOut.close()
      }
    }
  }

  test("Spark 3 writer keeps whole-iterator serialization behavior") {
    TaskResources.runUnsafe {
      val values = Seq(101L, 202L, 303L)
      val acknowledgements = ArrayBuffer.empty[Long]
      val batches = newBatches(values, acknowledgements)
      val bytes = new ByteArrayOutputStream()
      val dataOut = new DataOutputStream(bytes)
      val writer = new ColumnarArrowBatchStreamWriter(schema, "UTC")

      try {
        writer.writeAll(batches.iterator, dataOut)
        assert(acknowledgements == values)
        assert(readValues(bytes.toByteArray) == values)
      } finally {
        writer.close()
        batches.foreach(_.close())
        dataOut.close()
      }
    }
  }

  test("Spark 3 writer preserves empty-partition protocol") {
    TaskResources.runUnsafe {
      val bytes = new ByteArrayOutputStream()
      val dataOut = new DataOutputStream(bytes)
      val writer = new ColumnarArrowBatchStreamWriter(schema, "UTC")

      try {
        writer.writeAll(Iterator.empty, dataOut)
        assert(bytes.size() == 0)
      } finally {
        writer.close()
        dataOut.close()
      }
    }
  }

  private def newBatches(
      values: Seq[Long],
      acknowledgements: ArrayBuffer[Long]): Seq[ColumnarBatch] = {
    values.map {
      value =>
        val columns = ArrowWritableColumnVector.allocateColumns(1, schema)
        columns(0).putLong(0, value)
        columns(0).setValueCount(1)
        new AckingPythonInputBatch(
          columns.map(column => column: ColumnVector),
          1,
          () => acknowledgements += value)
    }
  }

  private def readValues(bytes: Array[Byte]): Seq[Long] = {
    val reader = new ArrowStreamReader(
      new ByteArrayInputStream(bytes),
      ArrowBufferAllocators.contextInstance())
    val values = ArrayBuffer.empty[Long]
    try {
      while (reader.loadNextBatch()) {
        val vector = reader.getVectorSchemaRoot.getVector(0).asInstanceOf[BigIntVector]
        (0 until reader.getVectorSchemaRoot.getRowCount).foreach {
          row => values += vector.get(row)
        }
      }
      values.toSeq
    } finally {
      reader.close(false)
    }
  }
}
