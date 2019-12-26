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

package org.apache.arrow.gandiva.evaluator;

import io.netty.buffer.ArrowBuf;
import org.apache.arrow.gandiva.exceptions.EvaluatorClosedException;
import org.apache.arrow.gandiva.exceptions.GandivaException;
import org.apache.arrow.gandiva.expression.ArrowTypeHelper;
import org.apache.arrow.gandiva.expression.Condition;
import org.apache.arrow.gandiva.ipc.GandivaTypes;
import org.apache.arrow.vector.ipc.message.ArrowBuffer;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class NestedLoopJoin {

    private static final Logger logger = LoggerFactory.getLogger(Filter.class);

    private final JniWrapper wrapper;
    private final long moduleId;
    private final Schema leftSchema;
    private final Schema rightSchema;
    private boolean closed;

    private NestedLoopJoin(JniWrapper wrapper, long moduleId, Schema leftSchema, Schema rightSchema) {
        this.wrapper = wrapper;
        this.moduleId = moduleId;
        this.leftSchema = leftSchema;
        this.rightSchema = rightSchema;
        this.closed = false;
    }

    public static NestedLoopJoin make(Schema leftSchema, Schema rightSchema, Condition condition) throws GandivaException {
        return make(leftSchema, rightSchema, condition, JniLoader.getDefaultConfiguration());
    }

    public static NestedLoopJoin make(Schema leftSchema, Schema rightSchema, Condition condition, long configurationId)
            throws GandivaException {
        // Invoke the JNI layer to create the LLVM NLJ module.
        GandivaTypes.Condition conditionBuf = condition.toProtobuf();
        GandivaTypes.Schema leftSchemaBuf = ArrowTypeHelper.arrowSchemaToProtobuf(leftSchema);
        GandivaTypes.Schema rightSchemaBuf = ArrowTypeHelper.arrowSchemaToProtobuf(rightSchema);
        JniWrapper wrapper = JniLoader.getInstance().getWrapper();
        long moduleId = wrapper.buildNLJ(leftSchemaBuf.toByteArray(), rightSchemaBuf.toByteArray(),
                conditionBuf.toByteArray(), configurationId);
        logger.debug("Created module for the nested loop join with id {}", moduleId);
        return new NestedLoopJoin(wrapper, moduleId, leftSchema, rightSchema);
    }

    public void evaluate(ArrowRecordBatch leftRecordBatch, ArrowRecordBatch rightRecordBatch, SelectionVector leftSelectionVector, SelectionVector rightSelectionVector)
            throws GandivaException {
        evaluate(leftRecordBatch.getLength(), leftRecordBatch.getBuffers(), leftRecordBatch.getBuffersLayout(),
                 rightRecordBatch.getLength(), rightRecordBatch.getBuffers(), rightRecordBatch.getBuffersLayout(),
                 leftSelectionVector, rightSelectionVector);
    }

    public void evaluate(int leftNumRows, List<ArrowBuf> leftBuffers, int rightNumRows, List<ArrowBuf> rightBuffers,
                         SelectionVector leftSelectionVector, SelectionVector rightSelectionVector) throws GandivaException {
        List<ArrowBuffer> leftBuffersLayout = new ArrayList<>();
        List<ArrowBuffer> rightBuffersLayout = new ArrayList<>();
        long offset = 0;
        for (ArrowBuf arrowBuf : leftBuffers) {
            long size = arrowBuf.readableBytes();
            leftBuffersLayout.add(new ArrowBuffer(offset, size));
            offset += size;
        }
        offset = 0;
        for (ArrowBuf arrowBuf : rightBuffers) {
            long size = arrowBuf.readableBytes();
            rightBuffersLayout.add(new ArrowBuffer(offset, size));
            offset += size;
        }
        evaluate(leftNumRows, leftBuffers, leftBuffersLayout, rightNumRows, rightBuffers, rightBuffersLayout, leftSelectionVector, rightSelectionVector);
    }

    private void evaluate(int leftNumRows, List<ArrowBuf> leftBuffers, List<ArrowBuffer> leftBuffersLayout,
                          int rightNumRows, List<ArrowBuf> rightBuffers, List<ArrowBuffer> rightBuffersLayout,
                          SelectionVector leftSelectionVector, SelectionVector rightSelectionVector) throws GandivaException {
        if (this.closed) {
            throw new EvaluatorClosedException();
        }
        if (leftSelectionVector.getMaxRecords() < leftNumRows * rightNumRows) {
            logger.error("left selectionVector has capacity for " + leftSelectionVector.getMaxRecords() +
                    " rows, minimum required " + leftNumRows * rightNumRows);
            throw new GandivaException("SelectionVector too small");
        }

        if (rightSelectionVector.getMaxRecords() < leftNumRows * rightNumRows) {
            logger.error("right selectionVector has capacity for " + rightSelectionVector.getMaxRecords() +
                    " rows, minimum required " + leftNumRows * rightNumRows);
            throw new GandivaException("SelectionVector too small");
        }

        long[] leftBufAddrs = new long[leftBuffers.size()];
        long[] leftBufSizes = new long[leftBuffers.size()];

        int idx = 0;
        for (ArrowBuf buf : leftBuffers) {
            leftBufAddrs[idx++] = buf.memoryAddress();
        }

        idx = 0;
        for (ArrowBuffer bufLayout : leftBuffersLayout) {
            leftBufSizes[idx++] = bufLayout.getSize();
        }

        long[] rightBufAddrs = new long[rightBuffers.size()];
        long[] rightBufSizes = new long[rightBuffers.size()];

        idx = 0;
        for (ArrowBuf buf : rightBuffers) {
            rightBufAddrs[idx++] = buf.memoryAddress();
        }

        idx = 0;
        for (ArrowBuffer bufLayout : rightBuffersLayout) {
            rightBufSizes[idx++] = bufLayout.getSize();
        }

        int numRecords = wrapper.evaluateNLJ(this.moduleId, leftNumRows,
                leftBufAddrs, leftBufSizes, rightNumRows,
                rightBufAddrs, rightBufSizes,
                leftSelectionVector.getType().getNumber(),
                leftSelectionVector.getBuffer().memoryAddress(), leftSelectionVector.getBuffer().capacity(),
                rightSelectionVector.getType().getNumber(),
                rightSelectionVector.getBuffer().memoryAddress(), rightSelectionVector.getBuffer().capacity());

        if (numRecords >= 0) {
            leftSelectionVector.setRecordCount(numRecords);
            rightSelectionVector.setRecordCount(numRecords);
        }
    }

    public void close() throws GandivaException {
        if (this.closed) {
            return;
        }

        wrapper.closeNLJ(this.moduleId);
        this.closed = true;
    }
}
