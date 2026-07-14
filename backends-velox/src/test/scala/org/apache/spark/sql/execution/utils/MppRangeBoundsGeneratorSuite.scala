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
package org.apache.spark.sql.execution.utils

import org.apache.spark.SparkFunSuite
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Ascending, AttributeReference, Descending, GenericInternalRow, NullsFirst, NullsLast, SortOrder, UnsafeProjection}
import org.apache.spark.sql.types.{DecimalType, DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.unsafe.types.UTF8String

import com.fasterxml.jackson.databind.ObjectMapper

class MppRangeBoundsGeneratorSuite extends SparkFunSuite {

  test("range descriptor preserves mixed directions, null ordering, and boundary values") {
    val intKey = AttributeReference("i", IntegerType, nullable = true)()
    val stringKey = AttributeReference("s", StringType, nullable = true)()
    val projection = UnsafeProjection.create(Seq(intKey, stringKey), Seq(intKey, stringKey))
    val rows: Array[InternalRow] = Array(
      projection(new GenericInternalRow(Array[Any](1, null))).copy(),
      projection(new GenericInternalRow(Array[Any](7, UTF8String.fromString("seven")))).copy())
    val ordering = Seq(
      SortOrder(intKey, Ascending, NullsFirst, Seq.empty),
      SortOrder(stringKey, Descending, NullsLast, Seq.empty))

    val descriptor = new ObjectMapper().readTree(MppRangeBoundsGenerator.encode(ordering, rows))

    assert(descriptor.get("version").asInt() == 1)
    assert(descriptor.get("keys").get(0).get("ascending").asBoolean())
    assert(descriptor.get("keys").get(0).get("nullsFirst").asBoolean())
    assert(!descriptor.get("keys").get(1).get("ascending").asBoolean())
    assert(!descriptor.get("keys").get(1).get("nullsFirst").asBoolean())
    assert(descriptor.get("bounds").size() == 2)
    assert(descriptor.get("bounds").get(0).get(0).get("value").asInt() == 1)
    assert(descriptor.get("bounds").get(0).get(1).get("isNull").asBoolean())
    assert(descriptor.get("bounds").get(1).get(1).get("value").asText() == "seven")
  }

  test("D1 type guard accepts verified keys and rejects unsupported ordering domains") {
    assert(MppRangeBoundsGenerator.supports(IntegerType))
    assert(MppRangeBoundsGenerator.supports(StringType))
    assert(!MppRangeBoundsGenerator.supports(DecimalType(18, 2)))
    assert(!MppRangeBoundsGenerator.supports(FloatType))
    assert(!MppRangeBoundsGenerator.supports(DoubleType))
  }
}
