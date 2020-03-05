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

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

#include "Arc/Arc.h"

using namespace mlir;
using namespace arc;
using namespace types;

#include "Arc/ArcOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ArcDialect
//===----------------------------------------------------------------------===//

ArcDialect::ArcDialect(mlir::MLIRContext *ctx) : mlir::Dialect("arc", ctx) {
  addOperations<
#define GET_OP_LIST
#include "Arc/Arc.cpp.inc"
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

//===----------------------------------------------------------------------===//
// ConstantIntOp
//===----------------------------------------------------------------------===//
static ParseResult parseConstantIntOp(OpAsmParser &parser,
                                      OperationState &state) {
  Attribute value;
  if (parser.parseAttribute(value, "value", state.attributes))
    return failure();

  Type type = value.getType();
  return parser.addTypeToList(type, state.types);
}

static void print(arc::ConstantIntOp constOp, OpAsmPrinter &printer) {
  printer << arc::ConstantIntOp::getOperationName() << ' ' << constOp.value();
}

static LogicalResult verify(arc::ConstantIntOp constOp) {
  auto opType = constOp.getType();
  auto value = constOp.value();
  auto valueType = value.getType();

  // ODS already generates checks to make sure the result type is
  // valid. We just need to additionally check that the value's
  // attribute type is consistent with the result type.
  switch (value.getKind()) {
  case StandardAttributes::Integer: {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } break;
  default:
    return constOp.emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

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
// General helpers for comparison ops, stolen from the standard dialect
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getCheckedI1SameShape(Type type) {
  auto i1Type = IntegerType::get(1, type.getContext());
  if (type.isIntOrIndexOrFloat())
    return i1Type;
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type);
  return Type();
}

static Type getI1SameShape(Type type) {
  Type res = getCheckedI1SameShape(type);
  assert(res && "expected type with valid i1 shape");
  return res;
}

/// Stolen from the standard dialect.
static void printArcBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op should have two operands");
  assert(op->getNumResults() == 1 && "binary op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (op->getOperand(0).getType() != resultType ||
      op->getOperand(1).getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  p << op->getName() << ' '
    << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Arc/Arc.cpp.inc"
