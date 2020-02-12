//===- Arc IR Dialect registration in MLIR --------------------------------===//
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

#include "arc/Types.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace arc;
using namespace types;

//===----------------------------------------------------------------------===//
// Shared Functions
//===----------------------------------------------------------------------===//

namespace arc::types {

bool isValueType(Type type) {
  switch (type.getKind()) {
  case StandardTypes::BF16:
  case StandardTypes::F16:
  case StandardTypes::F32:
  case StandardTypes::F64:
  case StandardTypes::Integer:
  case StandardTypes::Vector:
  case StandardTypes::RankedTensor:
  case StandardTypes::UnrankedTensor:
  case StandardTypes::UnrankedMemRef:
  case StandardTypes::MemRef:
  case StandardTypes::Complex:
  case StandardTypes::None:
    return true;
  case StandardTypes::Tuple:
    for (auto t : type.cast<TupleType>().getTypes())
      if (!isValueType(t))
        return false;
    return true;
  default:
    return false;
  }
}

bool isBuilderType(Type type) {
  switch (type.getKind()) {
  case Appender:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// AppenderType
//===----------------------------------------------------------------------===//

struct AppenderTypeStorage : public TypeStorage {
  AppenderTypeStorage(Type mergeType) : mergeType(mergeType) {}

  Type mergeType;

  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == KeyTy(mergeType); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static KeyTy getKey(Type mergeType) { return KeyTy(mergeType); }

  static AppenderTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<AppenderTypeStorage>())
        AppenderTypeStorage(key);
  }
};

Type AppenderType::getMergeType() { return getImpl()->mergeType; }

AppenderType AppenderType::get(Type mergeType) {
  mlir::MLIRContext *context = mergeType.getContext();
  return Base::get(context, Kind::Appender, mergeType);
}

Type AppenderType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::Type mergeType;
  if (parser.parseType(mergeType))
    return nullptr;
  if (!isValueType(mergeType)) {
    parser.emitError(loc,
                     "merge type for an appender must be a value type, got: ")
        << mergeType;
    return nullptr;
  }
  if (parser.parseGreater())
    return nullptr;
  return AppenderType::get(mergeType);
}

void AppenderType::print(DialectAsmPrinter &os) const {
  os << "appender" << '<' << t.getMergeType() << '>';
}
} // namespace arc::types
