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

#ifndef GANDIVA_JOIN_CACHE_KEY_H
#define GANDIVA_JOIN_CACHE_KEY_H

#include <memory>
#include <string>
#include <thread>

#include "boost/functional/hash.hpp"
#include "gandiva/arrow.h"
#include "gandiva/nested_loop_join.h"

namespace gandiva {
class JoinCacheKey {
 public:
  JoinCacheKey(SchemaPtr left_schema, SchemaPtr right_schema, std::shared_ptr<Configuration> configuration,
                 Expression& expression)
      : left_schema_(left_schema), right_schema_(right_schema), configuration_(configuration), uniqifier_(0) {
    static const int kSeedValue = 4;
    size_t result = kSeedValue;
    expression_as_string_ = expression.ToString();
    UpdateUniqifier(expression_as_string_);
    boost::hash_combine(result, expression_as_string_);
    boost::hash_combine(result, configuration);
    boost::hash_combine(result, left_schema_->ToString());
    boost::hash_combine(result, right_schema_->ToString());
    boost::hash_combine(result, uniqifier_);
    hash_code_ = result;
  }

  std::size_t Hash() const { return hash_code_; }

  bool operator==(const JoinCacheKey& other) const {
    // arrow schema does not overload equality operators.
    if (!(left_schema_->Equals(*other.left_schema().get(), true))) {
      return false;
    }

    if (!(right_schema_->Equals(*other.right_schema().get(), true))) {
      return false;
    }

    if (configuration_ != other.configuration_) {
      return false;
    }

    if (expression_as_string_ != other.expression_as_string_) {
      return false;
    }

    if (uniqifier_ != other.uniqifier_) {
      return false;
    }
    return true;
  }

  bool operator!=(const JoinCacheKey& other) const { return !(*this == other); }

  SchemaPtr left_schema() const { return left_schema_; }

  SchemaPtr right_schema() const { return right_schema_; }

  std::string ToString() const {
    std::stringstream ss;
    // indent, window, indent_size, null_rep and skip new lines.
    arrow::PrettyPrintOptions options{0, 10, 2, "null", true};
    DCHECK_OK(PrettyPrint(*left_schema_.get(), options, &ss));
    DCHECK_OK(PrettyPrint(*right_schema_.get(), options, &ss));

    ss << "Condition: [" << expression_as_string_ << "]";
    return ss.str();
  }

 private:
  void UpdateUniqifier(const std::string& expr) {
    // caching of expressions with re2 patterns causes lock contention. So, use
    // multiple instances to reduce contention.
    if (expr.find(" like(") != std::string::npos) {
      uniqifier_ = std::hash<std::thread::id>()(std::this_thread::get_id()) % 16;
    }
  }

  const SchemaPtr left_schema_;
  const SchemaPtr right_schema_;
  const std::shared_ptr<Configuration> configuration_;
  std::string expression_as_string_;
  size_t hash_code_;
  uint32_t uniqifier_;
};
}  // namespace gandiva
#endif  // GANDIVA_JOIN_CACHE_KEY_H
