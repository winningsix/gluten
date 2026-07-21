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
package org.apache.spark.sql.vectorized;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.catalyst.util.MapData;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.unsafe.types.BinaryView;
import org.apache.spark.unsafe.types.CalendarInterval;
import org.apache.spark.unsafe.types.TimestampNanosVal;
import org.apache.spark.unsafe.types.UTF8String;
import org.apache.spark.unsafe.types.VariantVal;

public final class ArrowColumnarArray extends ArrayData {
  private final ColumnarArray delegate;

  public ArrowColumnarArray(ColumnVector data, int offset, int length) {
    this.delegate = new ColumnarArray(data, offset, length);
  }

  @Override
  public int numElements() {
    return delegate.numElements();
  }

  @Override
  public ArrayData copy() {
    return delegate.copy();
  }

  @Override
  public Object[] array() {
    return delegate.array();
  }

  @Override
  public void setNullAt(int ordinal) {
    delegate.setNullAt(ordinal);
  }

  @Override
  public void update(int ordinal, Object value) {
    delegate.update(ordinal, value);
  }

  @Override
  public boolean isNullAt(int ordinal) {
    return delegate.isNullAt(ordinal);
  }

  @Override
  public boolean getBoolean(int ordinal) {
    return delegate.getBoolean(ordinal);
  }

  @Override
  public byte getByte(int ordinal) {
    return delegate.getByte(ordinal);
  }

  @Override
  public short getShort(int ordinal) {
    return delegate.getShort(ordinal);
  }

  @Override
  public int getInt(int ordinal) {
    return delegate.getInt(ordinal);
  }

  @Override
  public long getLong(int ordinal) {
    return delegate.getLong(ordinal);
  }

  @Override
  public float getFloat(int ordinal) {
    return delegate.getFloat(ordinal);
  }

  @Override
  public double getDouble(int ordinal) {
    return delegate.getDouble(ordinal);
  }

  @Override
  public Decimal getDecimal(int ordinal, int precision, int scale) {
    return delegate.getDecimal(ordinal, precision, scale);
  }

  @Override
  public UTF8String getUTF8String(int ordinal) {
    return delegate.getUTF8String(ordinal);
  }

  @Override
  public byte[] getBinary(int ordinal) {
    return delegate.getBinary(ordinal);
  }

  @Override
  public BinaryView getBinaryView(int ordinal) {
    return delegate.getBinaryView(ordinal);
  }

  @Override
  public CalendarInterval getInterval(int ordinal) {
    return delegate.getInterval(ordinal);
  }

  @Override
  public TimestampNanosVal getTimestampNTZNanos(int ordinal) {
    return delegate.getTimestampNTZNanos(ordinal);
  }

  @Override
  public TimestampNanosVal getTimestampLTZNanos(int ordinal) {
    return delegate.getTimestampLTZNanos(ordinal);
  }

  @Override
  public VariantVal getVariant(int ordinal) {
    return delegate.getVariant(ordinal);
  }

  @Override
  public InternalRow getStruct(int ordinal, int numFields) {
    return delegate.getStruct(ordinal, numFields);
  }

  @Override
  public ArrayData getArray(int ordinal) {
    return delegate.getArray(ordinal);
  }

  @Override
  public MapData getMap(int ordinal) {
    return delegate.getMap(ordinal);
  }

  @Override
  public Object get(int ordinal, DataType dataType) {
    return delegate.get(ordinal, dataType);
  }
}
