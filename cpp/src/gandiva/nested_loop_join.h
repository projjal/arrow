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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/status.h"

#include "gandiva/arrow.h"
#include "gandiva/configuration.h"
#include "gandiva/expression.h"
#include "gandiva/selection_vector.h"
#include "gandiva/visibility.h"

namespace gandiva {

class LLVMGenerator;

class GANDIVA_EXPORT NLJ {
public:
  NLJ(std::unique_ptr<LLVMGenerator> llvm_generator, SchemaPtr build_schema, SchemaPtr probe_schema,
    std::shared_ptr<Configuration> config);

  ~NLJ();

  static Status Make(SchemaPtr build_schema, SchemaPtr probe_schema, ConditionPtr condition,
                      SelectionVector::Mode left_selection_vector_mode, SelectionVector::Mode right_selection_vector_mode,
                     std::shared_ptr<NLJ>* nlj) {
    return Make(build_schema, probe_schema, condition, left_selection_vector_mode, right_selection_vector_mode, ConfigurationBuilder::DefaultConfiguration(), nlj);
  }

  static Status Make(SchemaPtr build_schema, SchemaPtr probe_schema, ConditionPtr condition,
                    SelectionVector::Mode left_selection_vector_mode, SelectionVector::Mode right_selection_vector_mode,
                     std::shared_ptr<Configuration> config,
                     std::shared_ptr<NLJ>* nlj);

  Status Evaluate(const arrow::RecordBatch& build_batch, const arrow::RecordBatch& probe_batch,
                  std::shared_ptr<SelectionVector> build_selection,
                  std::shared_ptr<SelectionVector> probe_selection);

private:
  const std::unique_ptr<LLVMGenerator> llvm_generator_;
  const SchemaPtr build_schema_;
  const SchemaPtr probe_schema_;
  const std::shared_ptr<Configuration> configuration_;
};

}  // namespace gandiva
