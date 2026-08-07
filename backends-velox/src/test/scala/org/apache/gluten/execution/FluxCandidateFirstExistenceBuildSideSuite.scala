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

import org.apache.gluten.substrait.SubstraitContext

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.expressions.{AttributeReference, EqualTo}
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}
import org.apache.spark.sql.catalyst.plans.ExistenceJoin
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{BooleanType, LongType}

import com.google.protobuf.StringValue
import io.substrait.proto.JoinRel

class FluxCandidateFirstExistenceBuildSideSuite extends QueryTest with SharedSparkSession {

  private val normalizeBuildSideKey = "spark.gluten.mpp.normalizeJoinBuildSide"
  private val candidateFirstPrefix = "_gluten_candidate_first_exists_"

  private case class TestJoin(
      join: ShuffledHashJoinExecTransformer,
      leftKey: AttributeReference,
      exists: AttributeReference)

  private def existenceJoin(existsName: String, buildSide: BuildSide = BuildRight): TestJoin = {
    val leftKey = AttributeReference("left_key", LongType, nullable = false)()
    val rightKey = AttributeReference("right_key", LongType, nullable = false)()
    val exists = AttributeReference(existsName, BooleanType, nullable = false)()
    val left = FluxExchangeSourceTransformer(0, Seq(leftKey))
    val right = FluxExchangeSourceTransformer(1, Seq(rightKey))
    TestJoin(
      ShuffledHashJoinExecTransformer(
        leftKeys = Seq(leftKey),
        rightKeys = Seq(rightKey),
        joinType = ExistenceJoin(exists),
        buildSide = buildSide,
        condition = Some(EqualTo(leftKey, rightKey)),
        left = left,
        right = right,
        isSkewJoin = false
      ),
      leftKey,
      exists
    )
  }

  private def normalize(join: ShuffledHashJoinExecTransformer): ShuffledHashJoinExecTransformer = {
    FluxNativeQueryExec(join, Seq.empty, Seq.empty)
      .normalizeFluxJoinBuildSide(join)
      .asInstanceOf[ShuffledHashJoinExecTransformer]
  }

  private def substraitJoin(join: ShuffledHashJoinExecTransformer): io.substrait.proto.JoinRel = {
    join
      .transform(new SubstraitContext)
      .root
      .toProtobuf
      .getProject
      .getInput
      .getJoin
  }

  test("private candidate-first existence marker becomes BuildLeft RIGHT_SEMI_PROJECT") {
    withSQLConf(normalizeBuildSideKey -> "true") {
      val testJoin = existenceJoin(candidateFirstPrefix + "q21")
      val normalized = normalize(testJoin.join)

      assert(normalized.buildSide == BuildLeft)
      assert(normalized.joinType == ExistenceJoin(testJoin.exists))
      assert(
        normalized.output.map(_.exprId) == Seq(testJoin.leftKey.exprId, testJoin.exists.exprId))
      assert(normalized.outputPartitioning == normalized.left.outputPartitioning)

      val joinRel = substraitJoin(normalized)
      assert(joinRel.getType == JoinRel.JoinType.JOIN_TYPE_RIGHT_SEMI)
      val parameters = joinRel.getAdvancedExtension.getOptimization.unpack(classOf[StringValue])
      assert(parameters.getValue.contains("isExistenceJoin=1"))
    }
  }

  test("ordinary and lookalike ExistenceJoin attributes keep their original build side") {
    withSQLConf(normalizeBuildSideKey -> "true") {
      Seq("exists", "_gluten_candidate_first_exist_q21", "GLUTEN_candidate_first_exists_q21")
        .foreach {
          existsName =>
            val original = existenceJoin(existsName)
            val normalized = normalize(original.join)

            assert(
              normalized.buildSide == BuildRight,
              s"ordinary ExistenceJoin $existsName must not be normalized")
            assert(normalized.joinType == ExistenceJoin(original.exists))
            assert(substraitJoin(normalized).getType == JoinRel.JoinType.JOIN_TYPE_LEFT_SEMI)
        }
    }
  }

  test("ordinary BuildLeft ExistenceJoin preserves left output partitioning") {
    val ordinary = existenceJoin("exists", BuildLeft).join
    assert(ordinary.outputPartitioning == ordinary.left.outputPartitioning)
    assert(normalize(ordinary).buildSide == BuildLeft)
  }

  test("candidate-first normalization is fixed point and respects the global disable switch") {
    val original = existenceJoin(candidateFirstPrefix + "fixed_point")
    withSQLConf(normalizeBuildSideKey -> "true") {
      val normalized = normalize(original.join)
      assert(normalized.buildSide == BuildLeft)
      assert(normalize(normalized).fastEquals(normalized))
    }
    withSQLConf(normalizeBuildSideKey -> "false") {
      assert(normalize(original.join).buildSide == BuildRight)
    }
  }

}
