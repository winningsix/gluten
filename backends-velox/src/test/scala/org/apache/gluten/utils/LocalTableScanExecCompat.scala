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
package org.apache.gluten.utils

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.LocalTableScanExec

/** Builds LocalTableScanExec across Spark 3.5 (two arguments) and Spark 4 (three arguments). */
object LocalTableScanExecCompat {
  def apply(output: Seq[Attribute], rows: Seq[InternalRow]): LocalTableScanExec = {
    val constructors = classOf[LocalTableScanExec].getConstructors
    val constructor = constructors
      .find(_.getParameterCount == 2)
      .orElse(constructors.find(_.getParameterCount == 3))
      .getOrElse {
        throw new IllegalStateException(
          s"Unsupported LocalTableScanExec constructor arities: " +
            constructors.map(_.getParameterCount).sorted.mkString(","))
      }
    val arguments = constructor.getParameterCount match {
      case 2 => Array[AnyRef](output.asInstanceOf[AnyRef], rows.asInstanceOf[AnyRef])
      case 3 =>
        Array[AnyRef](output.asInstanceOf[AnyRef], rows.asInstanceOf[AnyRef], None)
    }
    constructor.newInstance(arguments: _*).asInstanceOf[LocalTableScanExec]
  }
}
