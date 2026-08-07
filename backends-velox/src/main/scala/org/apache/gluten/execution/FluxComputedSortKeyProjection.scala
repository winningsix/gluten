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

import org.apache.gluten.extension.columnar.rewrite.PullOutPreProject

import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.{ProjectExec, SortExec, SparkPlan}

/** Materialize computed keys that reached an already-offloaded native sort. */
private[gluten] object FluxComputedSortKeyProjection {

  def rewrite(sort: SortExecTransformer): SparkPlan = {
    if (sort.sortOrder.forall(_.child.isInstanceOf[Attribute])) {
      return sort
    }

    val rowSort =
      copyMetadata(SortExec(sort.sortOrder, sort.global, sort.child, sort.testSpillFrequency), sort)

    // PullOutPreProject creates exactly PostProject -> Sort -> PreProject for a computed sort key.
    // Re-offload only those three newly created nodes. Returning the original native sort on an
    // unexpected shape keeps validation fail-closed instead of transforming an unrelated row plan.
    PullOutPreProject.rewrite(rowSort) match {
      case postProject: ProjectExec =>
        postProject.child match {
          case rewrittenSort: SortExec =>
            rewrittenSort.child match {
              case preProject: ProjectExec =>
                val nativePreProject = copyMetadata(
                  ProjectExecTransformer(preProject.projectList, preProject.child),
                  preProject)
                val nativeSort = copyMetadata(
                  SortExecTransformer(
                    rewrittenSort.sortOrder,
                    rewrittenSort.global,
                    nativePreProject,
                    rewrittenSort.testSpillFrequency),
                  rewrittenSort)
                copyMetadata(
                  ProjectExecTransformer(postProject.projectList, nativeSort),
                  postProject)
              case _ => sort
            }
          case _ => sort
        }
      case _ => sort
    }
  }

  private def copyMetadata[T <: SparkPlan](target: T, source: SparkPlan): T = {
    target.copyTagsFrom(source)
    source.logicalLink.foreach(target.setLogicalLink)
    target
  }
}
