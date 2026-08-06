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
package org.apache.gluten.config

import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.Paths

class AllVeloxConfiguration extends AnyFunSuite {
  private val glutenHome: String =
    AllGlutenConfiguration.getCodeSourceLocation(this.getClass).split("backends-velox")(0)
  private val markdown = Paths.get(glutenHome, "docs", "velox-configuration.md").toAbsolutePath

  test("Validate cache largest size class pages") {
    val entry = VeloxConfig.COLUMNAR_VELOX_CACHE_LARGEST_SIZE_CLASS_PAGES
    assert(entry.defaultValue.contains(256))
    Seq(256, 512, 1024, 2048).foreach {
      value => assert(entry.valueConverter(value.toString) == value)
    }
    Seq(128, 768, 4096).foreach {
      value => assertThrows[IllegalArgumentException](entry.valueConverter(value.toString))
    }
  }

  test("Validate contiguous cache entries") {
    val entry = VeloxConfig.COLUMNAR_VELOX_CACHE_CONTIGUOUS_ENTRIES
    assert(entry.defaultValue.contains(false))
    assert(entry.valueConverter("true"))
    assert(!entry.valueConverter("false"))
  }

  test("Check velox backend configs") {
    val builder = MarkdownBuilder(getClass.getName)

    builder ++=
      s"""
         |---
         |layout: page
         |title: Configuration
         |nav_order: 16
         |---
         |
         |"""

    builder ++=
      s"""
         |## Gluten Velox backend configurations
         |
         | Key | Default | Description
         | --- | --- | ---
         |"""

    VeloxConfig.allEntries
      .filter(_.isPublic)
      .filter(!_.isExperimental)
      .sortBy(_.key)
      .foreach {
        entry =>
          val dft = entry.defaultValueString.replace("<", "&lt;").replace(">", "&gt;")
          builder += Seq(s"${entry.key}", s"$dft", s"${entry.doc}")
            .mkString("|")
      }

    builder ++=
      s"""
         |## Gluten Velox backend *experimental* configurations
         |
         | Key | Default | Description
         | --- | --- | ---
         |"""

    VeloxConfig.allEntries
      .filter(_.isPublic)
      .filter(_.isExperimental)
      .sortBy(_.key)
      .foreach {
        entry =>
          val dft = entry.defaultValueString.replace("<", "&lt;").replace(">", "&gt;")
          builder += Seq(s"${entry.key}", s"$dft", s"${entry.doc}")
            .mkString("|")
      }

    AllGlutenConfiguration.verifyOrRegenerateGoldenFile(
      markdown,
      builder.toMarkdown,
      "dev/gen-all-config-docs.sh")
  }
}
