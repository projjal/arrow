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

#include <cstdint>

#include "arrow/util/basic_decimal.h"
#include "arrow/util/decimal.h"

namespace gandiva {

using arrow::BasicDecimal128;

/// Represents a 128-bit decimal value along with its precision and scale.
class BasicDecimalScalar128 {
 public:
  constexpr BasicDecimalScalar128(int64_t high_bits, uint64_t low_bits, int32_t precision,
                                  int32_t scale)
      : value_(high_bits, low_bits), precision_(precision), scale_(scale) {}

  constexpr BasicDecimalScalar128(const BasicDecimal128& value, int32_t precision,
                                  int32_t scale)
      : value_(value), precision_(precision), scale_(scale) {}

  constexpr BasicDecimalScalar128(int32_t precision, int32_t scale)
      : precision_(precision), scale_(scale) {}

  int32_t scale() const { return scale_; }

  int32_t precision() const { return precision_; }

  const BasicDecimal128& value() const { return value_; }

  virtual inline std::string ToString() const {
    arrow::Decimal128 dvalue(value());
    return dvalue.ToString(0) + "," + std::to_string(precision()) + "," +
           std::to_string(scale());
  }

 private:
  BasicDecimal128 value_;
  int32_t precision_;
  int32_t scale_;
};

inline bool operator==(const BasicDecimalScalar128& left,
                       const BasicDecimalScalar128& right) {
  return left.value() == right.value() && left.precision() == right.precision() &&
         left.scale() == right.scale();
}

inline BasicDecimalScalar128 operator-(const BasicDecimalScalar128& operand) {
  return BasicDecimalScalar128{-operand.value(), operand.precision(), operand.scale()};
}

inline std::ostream& operator<<(std::ostream& os, const BasicDecimalScalar128& dec) {
  os << dec.ToString();
  return os;
}

}  // namespace gandiva

namespace std {
    template<>
    struct hash<gandiva::BasicDecimalScalar128> {
        std::size_t operator()(gandiva::BasicDecimalScalar128 const& s) const noexcept {
            arrow::BasicDecimal128 dvalue(s.value());
            std::size_t h0 = std::hash<int64_t>{}(dvalue.high_bits());
            std::size_t h1 = std::hash<uint64_t>{}(dvalue.low_bits());

            std::size_t h2 = std::hash<int32_t>{}(s.precision());
            std::size_t h3 = std::hash<int32_t>{}(s.scale());

            return (((h0 ^ (h1 << 1) >> 1)^(h2 << 1) >> 1) ^ h3 << 1);
        }
    };
}