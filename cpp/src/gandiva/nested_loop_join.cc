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

#include "nested_loop_join.h"

#include <memory>
#include <utility>
#include <vector>

#include "gandiva/bitmap_accumulator.h"
#include "gandiva/cache.h"
#include "gandiva/condition.h"
#include "gandiva/expr_validator.h"
#include "gandiva/join_cache_key.h"
#include "gandiva/llvm_generator.h"

namespace gandiva {

NLJ::NLJ(std::unique_ptr<LLVMGenerator> llvm_generator, SchemaPtr build_schema,
                SchemaPtr probe_schema, std::shared_ptr<Configuration> configuration)
  : llvm_generator_(std::move(llvm_generator)),
    build_schema_(build_schema),
    probe_schema_(probe_schema),
    configuration_(configuration) {}

NLJ::~NLJ() {}

Status NLJ::Make(SchemaPtr probe_schema, SchemaPtr build_schema, ConditionPtr condition,
                  SelectionVector::Mode left_selection_vector_mode, SelectionVector::Mode right_selection_vector_mode,
                    std::shared_ptr<Configuration> configuration,
                    std::shared_ptr<NLJ>* nlj) {
  ARROW_RETURN_IF(build_schema == nullptr, Status::Invalid("Build Schema cannot be null"));
  ARROW_RETURN_IF(probe_schema == nullptr, Status::Invalid("Probe Schema cannot be null"));
  ARROW_RETURN_IF(condition == nullptr, Status::Invalid("Condition cannot be null"));
  ARROW_RETURN_IF(configuration == nullptr,
                  Status::Invalid("Configuration cannot be null"));

  static Cache<JoinCacheKey, std::shared_ptr<NLJ>> cache;
  JoinCacheKey cache_key(probe_schema, build_schema, configuration, *(condition.get()));
  auto cachedJoin = cache.GetModule(cache_key);
  if (cachedJoin != nullptr) {
    *nlj = cachedJoin;
    return Status::OK();
  }

  // Build LLVM generator, and generate code for the specified expression
  std::unique_ptr<LLVMGenerator> llvm_gen;
  ARROW_RETURN_NOT_OK(LLVMGenerator::Make(configuration, &llvm_gen));


  // Run the validation on the expression.
  // Return if the expression is invalid since we will not be able to process further.
  ExprValidator expr_validator(llvm_gen->types(), probe_schema, build_schema);
  ARROW_RETURN_NOT_OK(expr_validator.Validate(condition));

  ARROW_RETURN_NOT_OK(llvm_gen->Build(condition, left_selection_vector_mode, right_selection_vector_mode));

  *nlj = std::make_shared<NLJ>(std::move(llvm_gen), build_schema, probe_schema, configuration);
  cache.PutModule(cache_key, *nlj);

  return Status::OK();
}

Status NLJ::Evaluate(const arrow::RecordBatch& probe_batch, const arrow::RecordBatch& build_batch,
                        std::shared_ptr<SelectionVector> probe_selection,
                        std::shared_ptr<SelectionVector> build_selection) {
  const auto num_build_rows = build_batch.num_rows();
  const auto num_probe_rows = probe_batch.num_rows();

  ARROW_RETURN_IF(!build_batch.schema()->Equals(*build_schema_),
                  Status::Invalid("BuildBatch schema must be equal to expected schema"));
  ARROW_RETURN_IF(!probe_batch.schema()->Equals(*probe_schema_),
                  Status::Invalid("ProbeBatch schema must be equal to expected schema"));
  ARROW_RETURN_IF(num_build_rows == 0, Status::Invalid("BuildBatch must be non-empty."));
  ARROW_RETURN_IF(num_probe_rows == 0, Status::Invalid("ProbeBatch must be non-empty."));

  return llvm_generator_->ExecuteJoin(build_batch, probe_batch, build_selection, probe_selection);       
}

}  // namespace gandiva
