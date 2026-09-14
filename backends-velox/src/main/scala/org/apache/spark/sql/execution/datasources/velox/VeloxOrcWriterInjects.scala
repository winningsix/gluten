/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0.
 */
package org.apache.spark.sql.execution.datasources.velox

import org.apache.spark.sql.internal.SQLConf

import scala.collection.JavaConverters.mapAsJavaMapConverter

/** Native ORC write configuration consumed by the cuDF-backed table writer. */
class VeloxOrcWriterInjects extends VeloxFormatWriterInjects {
  override def nativeConf(
      options: Map[String, String],
      compressionCodec: String): java.util.Map[String, String] = {
    // The native option parser shares this compression key across its table
    // writers. The WriteRel format independently selects ORC versus Parquet.
    Map(
      SQLConf.PARQUET_COMPRESSION.key -> compressionCodec,
      SQLConf.SESSION_LOCAL_TIMEZONE.key -> options.getOrElse(
        SQLConf.SESSION_LOCAL_TIMEZONE.key,
        SQLConf.SESSION_LOCAL_TIMEZONE.defaultValueString)
    ).asJava
  }

  override val formatName: String = "orc"
}
