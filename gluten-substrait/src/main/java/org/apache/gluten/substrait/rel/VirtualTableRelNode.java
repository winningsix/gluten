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
package org.apache.gluten.substrait.rel;

import org.apache.gluten.substrait.expression.LiteralNode;
import org.apache.gluten.substrait.type.TypeNode;
import org.apache.gluten.utils.SubstraitUtil;

import io.substrait.proto.Expression;
import io.substrait.proto.NamedStruct;
import io.substrait.proto.ReadRel;
import io.substrait.proto.Rel;
import io.substrait.proto.RelCommon;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class VirtualTableRelNode implements RelNode, Serializable {
  private final List<TypeNode> types = new ArrayList<>();
  private final List<String> names = new ArrayList<>();
  private final List<List<LiteralNode>> rows = new ArrayList<>();

  VirtualTableRelNode(List<TypeNode> types, List<String> names, List<List<LiteralNode>> rows) {
    this.types.addAll(types);
    this.names.addAll(names);
    this.rows.addAll(rows);
  }

  @Override
  public Rel toProtobuf() {
    RelCommon.Builder relCommonBuilder = RelCommon.newBuilder();
    relCommonBuilder.setDirect(RelCommon.Direct.newBuilder());

    NamedStruct.Builder nStructBuilder =
        SubstraitUtil.createNameStructBuilder(types, names, new ArrayList<>());

    ReadRel.VirtualTable.Builder virtualTableBuilder = ReadRel.VirtualTable.newBuilder();
    for (List<LiteralNode> row : rows) {
      Expression.Literal.Struct.Builder rowBuilder = Expression.Literal.Struct.newBuilder();
      for (LiteralNode literal : row) {
        rowBuilder.addFields(literal.toProtobuf().getLiteral());
      }
      virtualTableBuilder.addValues(rowBuilder);
    }

    ReadRel.Builder readBuilder = ReadRel.newBuilder();
    readBuilder.setCommon(relCommonBuilder.build());
    readBuilder.setBaseSchema(nStructBuilder.build());
    readBuilder.setVirtualTable(virtualTableBuilder);

    Rel.Builder builder = Rel.newBuilder();
    builder.setRead(readBuilder.build());
    return builder.build();
  }
}
