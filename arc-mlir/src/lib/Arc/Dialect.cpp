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

#include "Arc/ArcDialect.h"
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
#include "Arc/ArcDialect.cpp.inc"
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

LogicalResult MakeVectorOp::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto ElemTy = Operation->getOperand(0).getType();
  auto TensorTy = Operation->getResult(0).getType().cast<TensorType>();
  if (!TensorTy.hasStaticShape())
    return emitOpError("result must have static shape: expected ")
           << RankedTensorType::get({NumOperands}, ElemTy);
  if (NumOperands != TensorTy.getNumElements())
    return emitOpError(
               "result does not match the number of operands: expected ")
           << TensorTy.getNumElements() << " but found " << NumOperands
           << " operands";
  return mlir::success();
}

LogicalResult MakeTupleOp::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto TupleTy = Operation->getResult(0).getType().cast<TupleType>();
  auto ElemTys = TupleTy.getTypes();
  if (NumOperands != TupleTy.size())
    return emitOpError(
               "result does not match the number of operands: expected ")
           << TupleTy.size() << " but found " << NumOperands << " operands";
  if (NumOperands == 0)
    return emitOpError("tuple must contain at least one element ");
  unsigned I = 0;
  for (const Type &ElemTy : Operation->getOperands().getTypes()) {
    if (ElemTys[I] != ElemTy)
      return emitOpError("operand types do not match: expected ")
             << ElemTys[I] << " but found " << ElemTy;
    I++;
  }
  return mlir::success();
}

LogicalResult IndexTupleOp::customVerify() {
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
           << Index << " does not match result: expected " << ResultTy
           << " but found " << IndexTy;
  return mlir::success();
}

LogicalResult IfOp::customVerify() {
  // We check that the result types of the blocks match the result
  // type of the operator.
  auto Op = this->getOperation();
  auto ResultTy = Op->getResult(0).getType();
  bool FoundErrors = false;
  auto CheckResultType = [this, ResultTy, &FoundErrors](ArcBlockResultOp R) {
    if (R.getResult().getType() != ResultTy) {
      FoundErrors = true;
      emitOpError(
          "result type does not match the type of the parent: expected ")
          << ResultTy << " but found " << R.getResult().getType();
    }
  };

  for (unsigned RegionIdx = 0; RegionIdx < Op->getNumRegions(); RegionIdx++)
    Op->getRegion(RegionIdx).walk(CheckResultType);

  if (FoundErrors)
    return mlir::failure();
  return mlir::success();
}

LogicalResult ArcYieldOp::customVerify() {
  auto Op = this->getOperation();
  auto ParentOp = Op->getParentOp();
  auto Results = ParentOp->getResults();
  auto Operands = Op->getOperands();

  if (isa<IfOp>(ParentOp) || isa<ForOp>(ParentOp)) {
    if (ParentOp->getNumResults() != Op->getNumOperands())
      return Op->emitOpError() << "parent of yield must have same number of "
                                  "results as the yield operands";
    for (auto e : llvm::zip(Results, Operands)) {
      if (std::get<0>(e).getType() != std::get<1>(e).getType())
        return Op->emitOpError()
               << "types mismatch between yield op and its parent";
    }
  } else {
    return Op->emitOpError()
           << "yield only terminates If, For or Parallel regions";
  }

  return success();
}

LogicalResult ForOp::customVerify() {
  auto Op = this->getOperation();
  auto &Func = Op->getRegion(0).front();
  auto BuilderTy = Op->getOperand(0).getType().cast<BuilderType>();
  auto NewBuilderTy = Op->getResult(0).getType().cast<BuilderType>();

  if (NewBuilderTy != BuilderTy)
    return emitOpError("result type does not match builder type, found: ")
           << NewBuilderTy << " but expected " << BuilderTy;

  auto NumArgs = Func.getNumArguments();

  if (NumArgs != 3)
    return emitOpError("block takes incorrect number of arguments, found: ")
           << NumArgs << " but expected " << 3;

  auto CollectionTy = Op->getOperand(1).getType().cast<TensorType>();
  auto ElemTy = CollectionTy.getElementType();

  auto Arg0 = Func.getArgument(0);
  auto Arg1 = Func.getArgument(1);
  auto Arg2 = Func.getArgument(2);
  auto ArgTy0 = Arg0.getType();
  auto ArgTy1 = Arg1.getType();
  auto ArgTy2 = Arg2.getType();

  if (!Arg0.hasOneUse())
    return emitOpError("block argument #0 must have exactly one use");

  if (ArgTy0 != BuilderTy)
    return emitOpError(
               "block argument #0 is different from builder type, found: ")
           << ArgTy0 << " but expected " << BuilderTy;

  if (!(ArgTy1.isa<IndexType>()))
    return emitOpError("block argument #1 is not an index type, found: ")
           << ArgTy1 << " but expected 'index'";

  if (ArgTy2 != ElemTy)
    return emitOpError("block argument #2 does not match element type of "
                       "collection, found: ")
           << ArgTy2 << " but expected " << ElemTy;

  return mlir::success();
}

LogicalResult MergeOp::customVerify() {
  auto Operation = this->getOperation();
  auto BuilderTy = Operation->getOperand(0).getType().cast<BuilderType>();
  auto BuilderMergeTy = BuilderTy.getMergeType();
  auto MergeTy = Operation->getOperand(1).getType();
  auto NewBuilderTy = Operation->getResult(0).getType().cast<BuilderType>();
  if (BuilderMergeTy != MergeTy)
    return emitOpError("operand type does not match merge type, found ")
           << MergeTy << " but expected " << BuilderMergeTy;
  if (BuilderTy != NewBuilderTy)
    return emitOpError("result type does not match builder type, found: ")
           << NewBuilderTy << " but expected " << BuilderTy;
  return mlir::success();
}

LogicalResult ResultOp::customVerify() {
  auto Operation = this->getOperation();
  auto BuilderTy = Operation->getOperand(0).getType().cast<BuilderType>();
  auto BuilderResultTy = BuilderTy.getResultType();
  auto ResultTy = Operation->getResult(0).getType().cast<BuilderType>();
  if (BuilderResultTy != ResultTy)
    return emitOpError("result type does not match that of builder, found ")
           << ResultTy << " but expected " << BuilderResultTy;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Arc/ArcDialect.cpp.inc"
