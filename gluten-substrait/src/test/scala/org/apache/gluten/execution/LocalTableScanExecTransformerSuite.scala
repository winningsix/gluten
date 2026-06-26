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

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, GenericInternalRow, UnsafeRow}
import org.apache.spark.sql.types.DateType

import org.scalatest.funsuite.AnyFunSuite

import java.io.{ByteArrayOutputStream, NotSerializableException, ObjectOutputStream}

class LocalTableScanExecTransformerSuite extends AnyFunSuite {

  test("materializes local rows before plan serialization") {
    val output = Seq(AttributeReference("snapshot_date", DateType, nullable = true)())
    val sourceRow = new NonSerializableBackedRow(Array[Any](1))

    intercept[NotSerializableException] {
      serialize(sourceRow)
    }

    val plan = LocalTableScanExecTransformer(output, Seq(sourceRow))

    assert(plan.rows.head.isInstanceOf[UnsafeRow])
    serialize(plan)
  }

  private def serialize(value: AnyRef): Unit = {
    val bytes = new ByteArrayOutputStream()
    val out = new ObjectOutputStream(bytes)
    try {
      out.writeObject(value)
    } finally {
      out.close()
    }
  }

  private class NonSerializableBackedRow(values: Array[Any]) extends GenericInternalRow(values) {
    private val nonSerializableObject = new Object

    override def copy(): GenericInternalRow = this
  }
}
