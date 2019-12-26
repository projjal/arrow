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

#ifndef GANDIVA_EXPR_ANNOTATOR_H
#define GANDIVA_EXPR_ANNOTATOR_H

#include <list>
#include <string>
#include <map>
#include <vector>

#include "gandiva/arrow.h"
#include "gandiva/eval_batch.h"
#include "gandiva/field_descriptor.h"
#include "gandiva/gandiva_aliases.h"
#include "gandiva/logging.h"
#include "gandiva/selection_vector.h"
#include "gandiva/visibility.h"

namespace gandiva {

/// \brief annotate the arrow fields in an expression, and use that
/// to convert the incoming arrow-format row batch to an EvalBatch.
class GANDIVA_EXPORT Annotator {
 public:
  Annotator() : buffer_count_(0), local_bitmap_count_(0) {}

  /// Add an annotated field descriptor for a field in an input schema.
  /// If the field is already annotated, returns that instead.
  FieldDescriptorPtr CheckAndAddInputFieldDescriptor(FieldPtr field, int ordinal=FieldDescriptor::kInvalidIdx);

  /// Add an annotated field descriptor for an output field.
  FieldDescriptorPtr AddOutputFieldDescriptor(FieldPtr field);

  int AddJoinOutput();

  /// Add a local bitmap (for saving validity bits of an intermediate node).
  /// Returns the index of the bitmap in the list of local bitmaps.
  int AddLocalBitMap() { return local_bitmap_count_++; }

  /// Prepare an eval batch for the incoming record batch.
  EvalBatchPtr PrepareEvalBatch(const arrow::RecordBatch& record_batch,
                                const ArrayDataVector& out_vector);

  EvalBatchPtr PrepareEvalBatch(const arrow::RecordBatch& left_batch,
                                const arrow::RecordBatch& right_batch,
                                const std::shared_ptr<SelectionVector> left_selection,
                                const std::shared_ptr<SelectionVector> right_selection);
  
  int buffer_count() { return buffer_count_; }

 private:
  /// Annotate a field and return the descriptor.
  FieldDescriptorPtr MakeDesc(FieldPtr field, bool is_output, int ordinal);

  /// Populate eval_batch by extracting the raw buffers from the arrow array, whose
  /// contents are represent by the annotated descriptor 'desc'.
  void PrepareBuffersForField(const FieldDescriptor& desc,
                              const arrow::ArrayData& array_data, EvalBatch* eval_batch,
                              bool is_output);

  void PrepareBuffersForSV(std::shared_ptr<SelectionVector> sv, EvalBatch* eval_batch, int idx);

  /// The list of input/output buffers (includes bitmap buffers, value buffers and
  /// offset buffers).
  int buffer_count_;

  /// The number of local bitmaps. These are used to save the validity bits for
  /// intermediate nodes in the expression tree.
  int local_bitmap_count_;

  /// map between field name and annotated input field descriptor.
  std::map<std::pair<std::string, int>, FieldDescriptorPtr> in_name_to_desc_;

  /// vector of annotated output field descriptors.
  std::vector<FieldDescriptorPtr> out_descs_;

  int out_idx_;
};

}  // namespace gandiva

#endif  // GANDIVA_EXPR_ANNOTATOR_H
