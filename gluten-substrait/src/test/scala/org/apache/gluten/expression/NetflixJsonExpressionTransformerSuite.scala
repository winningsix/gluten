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
package org.apache.gluten.expression

import org.apache.gluten.exception.GlutenNotSupportException
import org.apache.gluten.expression.fake.{NfJsonExtractScalar => FakeNfJsonExtractScalar}
import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.sql.catalyst.expressions.{AttributeReference, BoundReference, Expression, Literal}
import org.apache.spark.sql.types.{IntegerType, StringType}

import com.netflix.bdp.expressions.NfJsonExtractScalar
import org.scalatest.funsuite.AnyFunSuite

class NetflixJsonExpressionTransformerSuite extends AnyFunSuite {
  private val nullableJson = AttributeReference("json", StringType, nullable = true)()

  private def transformChild(expression: Expression): ExpressionTransformer = expression match {
    case literal: Literal => LiteralTransformer(literal)
    case attribute: AttributeReference =>
      AttributeReferenceTransformer(
        "selection",
        attribute,
        BoundReference(0, attribute.dataType, attribute.nullable))
    case other => fail(s"unexpected child in focused test: $other")
  }

  test("matches only the exact Netflix class and emits the bounded native function") {
    val expression = NfJsonExtractScalar(nullableJson, Literal("$.dump.parts[0].value"))
    val transformer = NetflixJsonExpressionTransformer
      .tryTransformFirstSlice(expression, transformChild)
      .get

    assert(transformer.substraitExprName === "nf_json_extract_scalar")
    assert(transformer.children.size === 2)
    assert(NetflixJsonExpressionTransformer.isExactFirstSliceDomain(expression))
    assert(
      !ExpressionConverter.canReplaceWithExpressionTransformer(expression, Seq(nullableJson)),
      "the bounded transformer must remain disconnected until native oracle parity passes"
    )

    val context = new SubstraitContext
    transformer.doTransform(context)
    assert(context.registeredFunction.keySet().toArray.exists {
      _.toString.startsWith("nf_json_extract_scalar:str_str")
    })

    val fake = FakeNfJsonExtractScalar(nullableJson, Literal("$.dump"))
    assert(NetflixJsonExpressionTransformer.tryTransformFirstSlice(fake, transformChild).isEmpty)
    assert(!NetflixJsonExpressionTransformer.isExactFirstSliceDomain(fake))
  }

  test("accepts the requested simple field and nonnegative-index grammar through depth sixteen") {
    val paths = Seq(
      "$.dump",
      "$.trackingInfo.videoId",
      "$.a[0]",
      "$.a_b.C9[2147483647]",
      "$.a.b[0].c[1].d[2].e[3].f[4].g[5].h[6].i")

    paths.foreach {
      path =>
        NetflixJsonExpressionTransformer.validateSimplePath(path)
        assert(
          NetflixJsonExpressionTransformer.isExactFirstSliceDomain(
            NfJsonExtractScalar(nullableJson, Literal(path))))
    }
  }

  test("fails closed for paths outside the exact first slice") {
    val unsupportedPaths = Seq(
      "$",
      "$[0]",
      "$.a[*]",
      "$.a[-1]",
      "$.a[00]",
      "$.a[2147483648]",
      "$.a[1:2]",
      "$.a[0,1]",
      "$.a[?(@.x)]",
      "$..a",
      "$['a']",
      "$.a.length()",
      "$.field-name",
      "$." + 0x00e9.toChar,
      "$.a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q"
    )

    unsupportedPaths.foreach {
      path =>
        val error = intercept[GlutenNotSupportException] {
          NetflixJsonExpressionTransformer.validateSimplePath(path)
        }
        assert(error.getMessage.contains("nf_json_extract_scalar"), path)
        assert(
          !NetflixJsonExpressionTransformer.isExactFirstSliceDomain(
            NfJsonExtractScalar(nullableJson, Literal(path))),
          path)
    }
  }

  test("requires STRING input, STRING output, and a non-null literal path") {
    val dynamicPath = AttributeReference("path", StringType, nullable = false)()
    val cases = Seq[Expression](
      NfJsonExtractScalar(nullableJson, dynamicPath),
      NfJsonExtractScalar(nullableJson, Literal.create(null, StringType)),
      NfJsonExtractScalar(
        AttributeReference("json", IntegerType, nullable = true)(),
        Literal("$.a"))
    )

    cases.foreach {
      expression =>
        val error = intercept[GlutenNotSupportException] {
          NetflixJsonExpressionTransformer.tryTransformFirstSlice(expression, transformChild)
        }
        assert(error.getMessage.contains("nf_json_extract_scalar"))
        assert(!NetflixJsonExpressionTransformer.isExactFirstSliceDomain(expression))
    }
  }
}
