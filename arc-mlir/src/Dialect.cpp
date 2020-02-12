//===- Arc IR Dialect registration in MLIR ------------------===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 RISE AB.
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
// This file implements the dialect for the Arc IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "arc/Dialect.h"

using namespace mlir;
using namespace arc;
using namespace types;

//===----------------------------------------------------------------------===//
// ArcDialect
//===----------------------------------------------------------------------===//
ArcDialect::ArcDialect(mlir::MLIRContext *ctx) : mlir::Dialect("arc", ctx) {
  addOperations<
#define GET_OP_LIST
#include "arc/Ops.cpp.inc"
      >();
  addTypes<arc::types::AppenderType>();
}

//===----------------------------------------------------------------------===//
// Arc Types
//===----------------------------------------------------------------------===//

bool arc::types::isValueType(Type type) {
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

bool arc::types::isBuilderType(Type type) {
  switch (type.getKind()) {
  case arc::types::Appender:
    return true;
  default:
    return false;
  }
}

struct arc::detail::AppenderTypeStorage : public TypeStorage {
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
  return Base::get(context, arc::types::Kind::Appender, mergeType);
}

//===----------------------------------------------------------------------===//
// Arc Type Parser
//===----------------------------------------------------------------------===//

Type parseAppenderType(DialectAsmParser &parser) {
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
  return arc::types::AppenderType::get(mergeType);
}

Type ArcDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (failed(parser.parseKeyword(&keyword)))
    return nullptr;
  if (keyword == "appender")
    return parseAppenderType(parser);
  parser.emitError(parser.getNameLoc(), "unknown type keyword " + keyword);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Arc Type Printer
//===----------------------------------------------------------------------===//

void ArcDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Arc type");
  case arc::types::Kind::Appender:
    arc::types::AppenderType t = type.cast<arc::types::AppenderType>();
    os << "appender" << '<' << t.getMergeType() << '>';
    break;
  }
}

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//

LogicalResult IfOp::customVerify() {
  // We check that the result types of the blocks matche the result
  // type of the operator.
  auto Op = this->getOperation();
  auto ResultTy = Op->getResult(0).getType();
  bool FoundErrors = false;
  auto CheckResultType = [this, ResultTy, &FoundErrors](ArcBlockResultOp R) {
    if (R.getResult().getType() != ResultTy) {
      FoundErrors = true;
      emitOpError("result type does not match the type of the parent: found ")
          << R.getResult().getType() << " but expected " << ResultTy;
    }
  };

  for (unsigned RegionIdx = 0; RegionIdx < Op->getNumRegions(); RegionIdx++)
    Op->getRegion(RegionIdx).walk(CheckResultType);

  if (FoundErrors)
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "arc/Ops.cpp.inc"
