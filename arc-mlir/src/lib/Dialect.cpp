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
// This file implements the dialect for the Arc IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "arc/Dialect.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

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
  addTypes<AppenderType>();
}

//===----------------------------------------------------------------------===//
// ArcDialect Type Parsing
//===----------------------------------------------------------------------===//

Type ArcDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (failed(parser.parseKeyword(&keyword)))
    return nullptr;
  if (keyword == "appender")
    return AppenderType::parse(parser);
  if (keyword == "unknown")
    return UnknownType::get(getContext());
  parser.emitError(parser.getNameLoc(), "unknown type keyword " + keyword);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ArcDialect Type Printing
//===----------------------------------------------------------------------===//

void ArcDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Arc type");
  case Appender:
    type.cast<AppenderType>().print(os);
    break;
  case Unknown:
    type.cast<UnknownType>().print(os);
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

void IfOp::inferReturnTypes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// Arc Type Inference
//===----------------------------------------------------------------------===//

LogicalResult MergeAppenderOp::inferReturnTypes(
    MLIRContext *ctx, Optional<Location> loc, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferedReturnTypes) {
  auto BuilderTy = operands[0].getType().cast<AppenderType>();
  auto MergeTy = BuilderTy.getMergeType();
  auto ValueTy = operands[1].getType();
  if (MergeTy != ValueTy)
    return emitOptionalError(
        loc,
        "'arc.merge_appender' op value type does not match merge type, found ",
        ValueTy, " but expected ", MergeTy);
  inferedReturnTypes.assign({BuilderTy});
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "arc/Ops.cpp.inc"
