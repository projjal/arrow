// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "gandiva/annotator.h"

#include <memory>
#include <string>

// #include "gandiva/field_descriptor.h"

namespace gandiva {

FieldDescriptorPtr Annotator::CheckAndAddInputFieldDescriptor(FieldPtr field, int ordinal) {
  // If the field is already in the map, return the entry.
  auto found = in_name_to_desc_.find(std::make_pair(field->name(), ordinal));
  if (found != in_name_to_desc_.end()) {
    return found->second;
  }

  auto desc = MakeDesc(field, false /*is_output*/, ordinal);
  in_name_to_desc_[std::make_pair(field->name(), ordinal)] = desc;
  return desc;
}

FieldDescriptorPtr Annotator::AddOutputFieldDescriptor(FieldPtr field) {
  auto desc = MakeDesc(field, true /*is_output*/, FieldDescriptor::kInvalidIdx);
  out_descs_.push_back(desc);
  return desc;
}

int Annotator::AddJoinOutput() {
  out_idx_ = buffer_count_;
  buffer_count_ += 2;
  return out_idx_;
}

FieldDescriptorPtr Annotator::MakeDesc(FieldPtr field, bool is_output, int ordinal) {
  int data_idx = buffer_count_++;
  int validity_idx = buffer_count_++;
  int offsets_idx = FieldDescriptor::kInvalidIdx;
  if (arrow::is_binary_like(field->type()->id())) {
    offsets_idx = buffer_count_++;
  }
  int data_buffer_ptr_idx = FieldDescriptor::kInvalidIdx;
  if (is_output) {
    data_buffer_ptr_idx = buffer_count_++;
  }
  return std::make_shared<FieldDescriptor>(field, data_idx, validity_idx, offsets_idx,
                                           data_buffer_ptr_idx, ordinal);
}

void Annotator::PrepareBuffersForField(const FieldDescriptor& desc,
                                       const arrow::ArrayData& array_data,
                                       EvalBatch* eval_batch, bool is_output) {
  int buffer_idx = 0;

  // The validity buffer is optional. Use nullptr if it does not have one.
  if (array_data.buffers[buffer_idx]) {
    uint8_t* validity_buf = const_cast<uint8_t*>(array_data.buffers[buffer_idx]->data());
    eval_batch->SetBuffer(desc.validity_idx(), validity_buf, array_data.offset);
  } else {
    eval_batch->SetBuffer(desc.validity_idx(), nullptr, array_data.offset);
  }
  ++buffer_idx;

  if (desc.HasOffsetsIdx()) {
    uint8_t* offsets_buf = const_cast<uint8_t*>(array_data.buffers[buffer_idx]->data());
    eval_batch->SetBuffer(desc.offsets_idx(), offsets_buf, array_data.offset);
    ++buffer_idx;
  }

  uint8_t* data_buf = const_cast<uint8_t*>(array_data.buffers[buffer_idx]->data());
  eval_batch->SetBuffer(desc.data_idx(), data_buf, array_data.offset);
  if (is_output) {
    // pass in the Buffer object for output data buffers. Can be used for resizing.
    uint8_t* data_buf_ptr =
        reinterpret_cast<uint8_t*>(array_data.buffers[buffer_idx].get());
    eval_batch->SetBuffer(desc.data_buffer_ptr_idx(), data_buf_ptr, array_data.offset);
  }
}

// sv descriptor?
void Annotator::PrepareBuffersForSV(std::shared_ptr<SelectionVector> sv, EvalBatch* eval_batch, int idx) {
  uint8_t* buf_ptr = const_cast<uint8_t*>(sv->GetBuffer().data());
  eval_batch->SetBuffer(idx, buf_ptr, 0 /*TODO*/);
}

EvalBatchPtr Annotator::PrepareEvalBatch(const arrow::RecordBatch& record_batch,
                                         const ArrayDataVector& out_vector) {
  EvalBatchPtr eval_batch = std::make_shared<EvalBatch>(
      record_batch.num_rows(), buffer_count_, local_bitmap_count_);

  // Fill in the entries for the input fields.
  for (int i = 0; i < record_batch.num_columns(); ++i) {
    const std::string& name = record_batch.column_name(i);
    auto found = in_name_to_desc_.find(std::make_pair(name, -1));
    if (found == in_name_to_desc_.end()) {
      // skip columns not involved in the expression.
      continue;
    }

    PrepareBuffersForField(*(found->second), *(record_batch.column(i))->data(),
                           eval_batch.get(), false /*is_output*/);
  }

  // Fill in the entries for the output fields.
  int idx = 0;
  for (auto& arraydata : out_vector) {
    const FieldDescriptorPtr& desc = out_descs_.at(idx);
    PrepareBuffersForField(*desc, *arraydata, eval_batch.get(), true /*is_output*/);
    ++idx;
  }
  return eval_batch;
}

EvalBatchPtr Annotator::PrepareEvalBatch(const arrow::RecordBatch& left_batch,
                                         const arrow::RecordBatch& right_batch,
                                         const std::shared_ptr<SelectionVector> left_selection,
                                         const std::shared_ptr<SelectionVector> right_selection) {
  int left_nrecords = left_batch.num_rows();
  int right_nrecords = right_batch.num_rows();
  int out_nrecords = left_nrecords * right_nrecords; // TODO Get this from config
  EvalBatchPtr eval_batch = std::make_shared<EvalBatch>(
      left_nrecords, right_nrecords, out_nrecords, buffer_count_ , local_bitmap_count_);

  // prepare buffers for left_batch
  for (int i = 0; i < left_batch.num_columns(); ++i) {
    const std::string& name = left_batch.column_name(i);
    auto found = in_name_to_desc_.find(std::make_pair(name, 0));
    if (found == in_name_to_desc_.end()) {
      // skip columns not involved in the expression.
      continue;
    }

    PrepareBuffersForField(*(found->second), *(left_batch.column(i))->data(),
                           eval_batch.get(), false /*is_output*/);
  }

  // prepare buffers for right_batch
  for (int i = 0; i < right_batch.num_columns(); ++i) {
    const std::string& name = right_batch.column_name(i);
    auto found = in_name_to_desc_.find(std::make_pair(name, 1));
    if (found == in_name_to_desc_.end()) {
      // skip columns not involved in the expression.
      continue;
    }

    PrepareBuffersForField(*(found->second), *(right_batch.column(i))->data(),
                           eval_batch.get(), false /*is_output*/);
  }

  PrepareBuffersForSV(left_selection, eval_batch.get(), out_idx_);
  PrepareBuffersForSV(right_selection, eval_batch.get(), out_idx_ + 1);
  return eval_batch;
}

}  // namespace gandiva
