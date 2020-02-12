//===- Dialect definition for the Arc IR ----------------------------------===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 KTH Royal Institute of Technology.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines the types of the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPES_H_
#define ARC_TYPES_H_

#include <mlir/IR/StandardTypes.h>

using namespace mlir;

namespace arc::types {

//===----------------------------------------------------------------------===//
// Arc Type Kinds
//===----------------------------------------------------------------------===//

enum Kind {
  Appender = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};

//===----------------------------------------------------------------------===//
// Arc Type Functions
//===----------------------------------------------------------------------===//

bool isValueType(Type type);
bool isBuilderType(Type type);

//===----------------------------------------------------------------------===//
// Arc Type Storages
//===----------------------------------------------------------------------===//

struct AppenderTypeStorage;

//===----------------------------------------------------------------------===//
// Arc Types
//===----------------------------------------------------------------------===//

class AppenderType
    : public Type::TypeBase<AppenderType, Type, AppenderTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == Appender; }

  static AppenderType get(Type mergeType);

  Type getMergeType();
};
} // namespace arc::types

#endif // ARC_TYPES_H_
