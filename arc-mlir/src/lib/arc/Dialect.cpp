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
  }
}

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//

LogicalResult MakeVector::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto ElemTy = Operation->getOperand(0).getType();
  auto TensorTy = Operation->getResult(0).getType().cast<TensorType>();
  if (!TensorTy.hasStaticShape())
    return emitOpError("result must have static shape, expected ")
           << RankedTensorType::get({NumOperands}, ElemTy);
  if (NumOperands != TensorTy.getNumElements())
    return emitOpError("result does not match the number of operands: found ")
           << NumOperands << " but expected " << TensorTy.getNumElements()
           << " operands";
  return mlir::success();
}

LogicalResult MakeTuple::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto TupleTy = Operation->getResult(0).getType().cast<TupleType>();
  auto ElemTys = TupleTy.getTypes();
  if (NumOperands != TupleTy.size())
    return emitOpError("result does not match the number of operands: found ")
           << NumOperands << " but expected " << TupleTy.size() << " operands";
  if (NumOperands == 0)
    return emitOpError("tuple must contain at least one element ");
  unsigned I = 0;
  for (const Type &ElemTy : Operation->getOperands().getTypes()) {
    if (ElemTys[I] != ElemTy)
      return emitOpError("operand types do not match, found ")
             << ElemTy << " but expected " << ElemTys[I];
    I++;
  }
  return mlir::success();
}

LogicalResult IndexTuple::customVerify() {
  auto Operation = this->getOperation();
  auto ResultTy = Operation->getResult(0).getType();
  auto TupleTy = Operation->getOperand(0).getType().cast<TupleType>();
  auto Index = getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
  auto NumElems = TupleTy.size();
  if (Index >= NumElems)
    return emitOpError("index ")
           << Index << " is out-of-bounds for tuple with size " << NumElems;
  auto ElemTys = TupleTy.getTypes();
  auto IndexTy = ElemTys[Index];
  if (IndexTy != ResultTy)
    return emitOpError("element type at index ")
           << Index << " does not match result, found " << IndexTy
           << " but expected " << ResultTy;
  return mlir::success();
}

LogicalResult If::customVerify() {
  // We check that the result types of the blocks matche the result
  // type of the operator.
  auto Op = this->getOperation();
  auto ResultTy = Op->getResult(0).getType();
  bool FoundErrors = false;
  auto CheckResultType = [this, ResultTy, &FoundErrors](ArcBlockResult R) {
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
