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
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Matchers.h>

#include "Arc/Arc.h"

using namespace mlir;
using namespace arc;
using namespace types;

#include "Arc/ArcOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ArcDialect
//===----------------------------------------------------------------------===//

void ArcDialect::initialize(void) {
  addOperations<
#define GET_OP_LIST
#include "Arc/Arc.cpp.inc"
      >();
  addTypes<AppenderType>();
  addTypes<ArconValueType>();
  addTypes<StreamType>();
  addTypes<StructType>();
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
  if (keyword == "arcon.value")
    return ArconValueType::parse(parser);
  if (keyword == "stream")
    return StreamType::parse(parser);
  if (keyword == "struct")
    return StructType::parse(parser);
  parser.emitError(parser.getNameLoc(), "unknown type keyword " + keyword);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ArcDialect Type Printing
//===----------------------------------------------------------------------===//

void ArcDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto t = type.dyn_cast<AppenderType>())
    t.print(os);
  else if (auto t = type.dyn_cast<ArconValueType>())
    t.print(os);
  else if (auto t = type.dyn_cast<StreamType>())
    t.print(os);
  else if (auto t = type.dyn_cast<StructType>())
    t.print(os);
  else
    llvm_unreachable("Unhandled Arc type");
}

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

// Stolen from standard dialect
OpFoldResult arc::AndOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// and(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

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
  if (value.isa<IntegerAttr>()) {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } else {
    return constOp.emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

OpFoldResult ConstantIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type. Stolen from the standard dialect.
Operation *ArcDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (type.isSignedInteger() || type.isUnsignedInteger())
    return builder.create<ConstantIntOp>(loc, type, value);
  return nullptr; // Let the standard dialect handle this
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
// comparison predicates.
static bool applyCmpPredicate(Arc_CmpIPredicate predicate, bool isUnsigned,
                              const APInt &lhs, const APInt &rhs) {
  switch (predicate) {
  case Arc_CmpIPredicate::eq:
    return lhs.eq(rhs);
  case Arc_CmpIPredicate::ne:
    return lhs.ne(rhs);
  case Arc_CmpIPredicate::lt:
    return isUnsigned ? lhs.ult(rhs) : lhs.slt(rhs);
  case Arc_CmpIPredicate::le:
    return isUnsigned ? lhs.ule(rhs) : lhs.sle(rhs);
  case Arc_CmpIPredicate::gt:
    return isUnsigned ? lhs.ugt(rhs) : lhs.sgt(rhs);
  case Arc_CmpIPredicate::ge:
    return isUnsigned ? lhs.uge(rhs) : lhs.sge(rhs);
  }
  llvm_unreachable("unknown comparison predicate");
}

// Constant folding hook for comparisons.
OpFoldResult arc::CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};
  bool isUnsigned = operands[0].getType().isUnsignedInteger();
  auto val = applyCmpPredicate(getPredicate(), isUnsigned, lhs.getValue(),
                               rhs.getValue());
  return IntegerAttr::get(IntegerType::get(getContext(), 1), APInt(1, val));
}

//===----------------------------------------------------------------------===//
// DivIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from standard dialect

OpFoldResult DivIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  bool isUnsigned = getType().isUnsignedInteger();
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    if (isUnsigned)
      return a.udiv(b);
    return a.sdiv_ov(b, overflowOrDiv0);
  });
  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// EmitOp
//===----------------------------------------------------------------------===//
LogicalResult EmitOp::customVerify() {
  auto Operation = this->getOperation();
  auto ElemTy = Operation->getOperand(0).getType();
  auto StreamTy =
      Operation->getOperand(1).getType().cast<StreamType>().getType();
  if (ElemTy != StreamTy)
    return emitOpError("Can't emit element of type ")
           << ElemTy << " on stream of " << StreamTy;
  return mlir::success();
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

LogicalResult MakeStructOp::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto StructTy = Operation->getResult(0).getType().cast<StructType>();
  auto FieldTys = StructTy.getFields();
  if (NumOperands != StructTy.getNumFields())
    return emitOpError("expected ")
           << StructTy.getNumFields() << " fields, but found " << NumOperands;
  unsigned I = 0;
  for (const Type &ElemTy : Operation->getOperands().getTypes()) {
    if (FieldTys[I].second != ElemTy)
      return emitOpError("operand types do not match: expected ")
             << FieldTys[I].second << " but found " << ElemTy;
    I++;
  }
  return mlir::success();
}

LogicalResult MakeTensorOp::customVerify() {
  auto Operation = this->getOperation();
  auto NumOperands = Operation->getNumOperands();
  auto TensorTy = Operation->getResult(0).getType().cast<RankedTensorType>();
  ArrayRef<int64_t> Shape = TensorTy.getShape();

  int64_t NoofElems = 1;

  for (int64_t n : Shape)
    NoofElems *= n;

  if (NumOperands != NoofElems)
    return emitOpError("wrong number of operands: expected ")
           << NoofElems << " but found " << NumOperands << " operands";
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
  auto Index =
      (*this)->getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
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
    if (R.getOperand().getType() != ResultTy) {
      FoundErrors = true;
      emitOpError(
          "result type does not match the type of the parent: expected ")
          << ResultTy << " but found " << R.getOperand().getType();
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

LogicalResult StructAccessOp::customVerify() {
  auto Operation = this->getOperation();
  auto ResultTy = Operation->getResult(0).getType();
  auto StructTy = Operation->getOperand(0).getType().cast<StructType>();
  auto Field = (*this)->getAttrOfType<StringAttr>("field").getValue();
  for (auto &i : StructTy.getFields())
    if (i.first.getValue().equals(Field)) {
      if (i.second == ResultTy)
        return mlir::success();
      else
        return emitOpError("field '")
               << Field << "' does not have a matching type, expected "
               << ResultTy << " but found " << i.second;
    }
  return emitOpError("field '") << Field << "' does not exist in " << StructTy;
}

OpFoldResult AddIOp::fold(ArrayRef<Attribute> operands) {
  /// addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  bool isUnsigned = getType().isUnsignedInteger();
  bool overflowDetected = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowDetected)
      return a;
    if (isUnsigned)
      return a.uadd_ov(b, overflowDetected);
    return a.sadd_ov(b, overflowDetected);
  });
  return overflowDetected ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from the standard dialect
OpFoldResult MulIOp::fold(ArrayRef<Attribute> operands) {
  /// muli(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// muli(x, 1) -> x
  if (matchPattern(rhs(), m_One()))
    return getOperand(0);

  bool isUnsigned = getType().isUnsignedInteger();

  // Don't fold if it would overflow
  bool overflow = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflow || !b) {
      overflow = true;
      return a;
    }
    if (isUnsigned)
      return a.umul_ov(b, overflow);
    return a.smul_ov(b, overflow);
  });
  return overflow ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult arc::OrOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// or(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// RemIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from standard dialect
OpFoldResult RemIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "remi_unsigned takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  bool isUnsigned = getType().isUnsignedInteger();
  return IntegerAttr::get(lhs.getType(), isUnsigned
                                             ? lhs.getValue().urem(rhsValue)
                                             : lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Stolen from standard dialect
OpFoldResult arc::SelectOp::fold(ArrayRef<Attribute> operands) {
  auto condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return getTrueValue();

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return getFalseValue();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from the standard dialect
OpFoldResult SubIOp::fold(ArrayRef<Attribute> operands) {
  /// addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  bool isUnsigned = getType().isUnsignedInteger();
  bool overflowDetected = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowDetected)
      return a;
    if (isUnsigned)
      return a.usub_ov(b, overflowDetected);
    return a.ssub_ov(b, overflowDetected);
  });
  return overflowDetected ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

// Stolen from standard dialect
OpFoldResult arc::XOrOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// xor(x,x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops, stolen from the standard dialect
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getCheckedI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
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

  p << op->getName() << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Arc/Arc.cpp.inc"
