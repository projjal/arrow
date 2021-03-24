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

package org.apache.arrow.gandiva.expression;

import java.nio.charset.Charset;
import java.util.Set;

import org.apache.arrow.gandiva.exceptions.GandivaException;
import org.apache.arrow.gandiva.ipc.GandivaTypes;

import com.google.protobuf.ByteString;

/**
 * In Node representation in java.
 */
public class InNode implements TreeNode {
  private static final Charset charset = Charset.forName("UTF-8");

  private final Set<Integer> intValues;
  private final Set<Long> longValues;
  private final Set<Decimal> decimalValues;
  private final Set<String> stringValues;
  private final Set<byte[]> binaryValues;
  private final TreeNode input;

  private InNode(Set<Integer> values, Set<Long> longValues, Set<String> stringValues, Set<byte[]>
          binaryValues, Set<Decimal> decimalValues, TreeNode node) {
    this.intValues = values;
    this.longValues = longValues;
    this.decimalValues = decimalValues;
    this.stringValues = stringValues;
    this.binaryValues = binaryValues;
    this.input = node;
  }

  public static InNode makeIntInExpr(TreeNode node, Set<Integer> intValues) {
    return new InNode(intValues, null, null, null, null, node);
  }

  public static InNode makeLongInExpr(TreeNode node, Set<Long> longValues) {
    return new InNode(null, longValues, null, null, null, node);
  }

  public static InNode makeDecimalInExpr(TreeNode node, Set<Decimal> decimalValues) {
    return new InNode(null, null, null, null, decimalValues, node);
  }

  public static InNode makeStringInExpr(TreeNode node, Set<String> stringValues) {
    return new InNode(null, null, stringValues, null, null, node);
  }

  public static InNode makeBinaryInExpr(TreeNode node, Set<byte[]> binaryValues) {
    return new InNode(null, null, null, binaryValues, null, node);
  }

  @Override
  public GandivaTypes.TreeNode toProtobuf() throws GandivaException {
    GandivaTypes.InNode.Builder inNode = GandivaTypes.InNode.newBuilder();

    inNode.setNode(input.toProtobuf());

    if (intValues != null) {
      GandivaTypes.IntConstants.Builder intConstants = GandivaTypes.IntConstants.newBuilder();
      intValues.stream().forEach(val -> intConstants.addIntValues(GandivaTypes.IntNode.newBuilder()
              .setValue(val).build()));
      inNode.setIntValues(intConstants.build());
    } else if (longValues != null) {
      GandivaTypes.LongConstants.Builder longConstants = GandivaTypes.LongConstants.newBuilder();
      longValues.stream().forEach(val -> longConstants.addLongValues(GandivaTypes.LongNode.newBuilder()
              .setValue(val).build()));
      inNode.setLongValues(longConstants.build());
    } else if (decimalValues != null) {
      GandivaTypes.DecimalConstants.Builder decimalConstants = GandivaTypes.DecimalConstants.newBuilder();
      decimalValues.stream().forEach(val -> decimalConstants.addDecimalValues(GandivaTypes.DecimalNode.newBuilder()
              .setValue(val).build()));
      inNode.setDecimalValues(decimalConstants.build());
    }else if (stringValues != null) {
      GandivaTypes.StringConstants.Builder stringConstants = GandivaTypes.StringConstants
              .newBuilder();
      stringValues.stream().forEach(val -> stringConstants.addStringValues(GandivaTypes.StringNode
              .newBuilder().setValue(ByteString.copyFrom(val.getBytes(charset))).build()));
      inNode.setStringValues(stringConstants.build());
    } else if (binaryValues != null) {
      GandivaTypes.BinaryConstants.Builder binaryConstants = GandivaTypes.BinaryConstants
              .newBuilder();
      binaryValues.stream().forEach(val -> binaryConstants.addBinaryValues(GandivaTypes.BinaryNode
              .newBuilder().setValue(ByteString.copyFrom(val)).build()));
      inNode.setBinaryValues(binaryConstants.build());
    }
    GandivaTypes.TreeNode.Builder builder = GandivaTypes.TreeNode.newBuilder();
    builder.setInNode(inNode.build());
    return builder.build();

  }
}
