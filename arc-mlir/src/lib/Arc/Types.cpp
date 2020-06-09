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

#include "Arc/Types.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace arc;
using namespace types;

namespace arc {
namespace types {

//===----------------------------------------------------------------------===//
// Shared Functions
//===----------------------------------------------------------------------===//

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
// BuilderType
//===----------------------------------------------------------------------===//

struct BuilderTypeStorage : public TypeStorage {
  BuilderTypeStorage(Type mergeTy, Type resultTy, unsigned subclassData = 0)
      : TypeStorage(subclassData), mergeType(mergeTy), resultType(resultTy) {}

  using KeyTy = std::pair<Type, Type>;

  bool operator==(const KeyTy &key) const {
    return key.first == mergeType && key.second == resultType;
  }

  Type mergeType;
  Type resultType;
};

Type BuilderType::getMergeType() const {
  return static_cast<ImplType *>(impl)->mergeType;
}

Type BuilderType::getResultType() const {
  return static_cast<ImplType *>(impl)->resultType;
}

//===----------------------------------------------------------------------===//
// AppenderType
//===----------------------------------------------------------------------===//

struct AppenderTypeStorage : public BuilderTypeStorage {
  AppenderTypeStorage(Type mergeTy, RankedTensorType resultTy)
      : BuilderTypeStorage(mergeTy, resultTy) {}

  using KeyTy = std::pair<Type, RankedTensorType>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static KeyTy getKey(Type mergeType, RankedTensorType resultType) {
    return KeyTy(mergeType, resultType);
  }

  static AppenderTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<AppenderTypeStorage>())
        AppenderTypeStorage(key.first, key.second);
  }
};

AppenderType AppenderType::get(Type mergeType, RankedTensorType resultType) {
  return Base::get(mergeType.getContext(), Appender, mergeType, resultType);
}

AppenderType AppenderType::getChecked(Type mergeType,
                                      RankedTensorType resultType,
                                      Location loc) {
  return Base::getChecked(loc, Appender, mergeType, resultType);
}

Type AppenderType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::Type mergeType;
  if (parser.parseType(mergeType))
    return nullptr;
  if (parser.parseGreater())
    return nullptr;
  auto resultType = RankedTensorType::getChecked({}, mergeType, loc);
  return AppenderType::getChecked(mergeType, resultType, loc);
}

void AppenderType::print(DialectAsmPrinter &os) const {
  os << "appender" << '<' << getMergeType() << '>';
}

LogicalResult
AppenderType::verifyConstructionInvariants(Location loc, Type mergeType,
                                           RankedTensorType resultType) {
  if (!isValueType(mergeType)) {
    emitOptionalError(loc, "appender merge type must be a value type: found ",
                      mergeType);
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<StructType::FieldTy>;

  StructTypeStorage(llvm::ArrayRef<StructType::FieldTy> elementTypes)
      : fields(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == fields; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<StructType::FieldTy> elementTypes) {
    return KeyTy(elementTypes);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<StructType::FieldTy> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  llvm::ArrayRef<StructType::FieldTy> fields;
};

StructType StructType::get(llvm::ArrayRef<StructType::FieldTy> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  mlir::MLIRContext *ctx = elementTypes.front().second.getContext();
  return Base::get(ctx, arc::types::Struct, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<StructType::FieldTy> StructType::getFields() const {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->fields;
}

size_t StructType::getNumFields() const { return getFields().size(); }

Type StructType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;
  Builder &builder = parser.getBuilder();

  SmallVector<StructType::FieldTy, 3> elementTypes;
  do {
    StringRef name;
    if (parser.parseKeyword(&name) || parser.parseColon())
      return nullptr;

    StructType::FieldTy elementType;
    elementType.first = StringAttr::get(name, builder.getContext());
    if (parser.parseType(elementType.second))
      return nullptr;

    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

void StructType::print(DialectAsmPrinter &os) const {
  // Print the struct type according to the parser format.
  os << "struct<";
  auto fields = getFields();
  for (unsigned i = 0; i < getNumFields(); i++) {
    if (i != 0)
      os << ", ";
    os << fields[i].first.getValue() << " : " << fields[i].second;
  }
  os << '>';
}

} // namespace types
} // namespace arc
