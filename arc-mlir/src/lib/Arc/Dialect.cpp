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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Matchers.h>

#include "Arc/Arc.h"

using namespace mlir;
using namespace arc;
using namespace types;

#include "Arc/ArcOpsDialect.cpp.inc"
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
  addTypes<ArconAppenderType>();
  addTypes<ArconMapType>();
  addTypes<ArconValueType>();
  addTypes<StreamType>();
  addTypes<EnumType>();
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
  if (keyword == "arcon.appender")
    return ArconAppenderType::parse(parser);
  if (keyword == "arcon.map")
    return ArconMapType::parse(parser);
  if (keyword == "arcon.value")
    return ArconValueType::parse(parser);
  if (keyword == "stream")
    return StreamType::parse(parser);
  if (keyword == "enum")
    return EnumType::parse(parser);
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
  else if (auto t = type.dyn_cast<ArconAppenderType>())
    t.print(os);
  else if (auto t = type.dyn_cast<ArconMapType>())
    t.print(os);
  else if (auto t = type.dyn_cast<ArconValueType>())
    t.print(os);
  else if (auto t = type.dyn_cast<StreamType>())
    t.print(os);
  else if (auto t = type.dyn_cast<EnumType>())
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

LogicalResult EnumAccessOp::customVerify() {
  auto ResultTy = result().getType();
  auto SourceTy = value().getType().cast<EnumType>();
  auto VariantTys = SourceTy.getVariants();
  auto WantedVariant = variant();

  // Check that the given type matches the specified variant.
  for (auto &i : VariantTys)
    if (i.first.getValue().equals(WantedVariant)) {
      if (i.second == ResultTy)
        return mlir::success();
      else
        return emitOpError(": variant '")
               << WantedVariant << "' does not have a matching type, expected "
               << ResultTy << " but found " << i.second;
    }
  return emitOpError(": variant '")
         << WantedVariant << "' does not exist in " << SourceTy;
}

LogicalResult EnumCheckOp::customVerify() {
  auto SourceTy = value().getType().cast<EnumType>();
  auto VariantTys = SourceTy.getVariants();
  auto WantedVariant = (*this)->getAttrOfType<StringAttr>("variant").getValue();

  // Check that the given type matches the specified variant.
  for (auto &i : VariantTys)
    if (i.first.getValue().equals(WantedVariant))
      return mlir::success();
  return emitOpError(": variant '")
         << WantedVariant << "' does not exist in " << SourceTy;
}

LogicalResult MakeEnumOp::customVerify() {
  auto ResultTy = result().getType().cast<EnumType>();
  Type SourceTy = NoneType::get(getContext());
  auto VariantTys = ResultTy.getVariants();
  auto WantedVariant = variant();

  if (values().size() > 1)
    return emitOpError(": only a single value expected");

  if (values().size())
    SourceTy = values()[0].getType();

  // Check that the given type matches the specified variant.
  for (auto &i : VariantTys)
    if (i.first.getValue().equals(WantedVariant)) {
      if (i.second == SourceTy)
        return mlir::success();
      else
        return emitOpError(": variant '")
               << WantedVariant << "' does not have a matching type, expected "
               << SourceTy << " but found " << i.second;
    }
  return emitOpError(": variant '")
         << WantedVariant << "' does not exist in " << ResultTy;
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

LogicalResult ArcBlockResultOp::customVerify() {
  // Check that our type matches the type of our parent if-op
  auto Op = this->getOperation();
  auto Parent = Op->getParentOp();

  if (Op->getNumOperands() > 1) {
    emitOpError("cannot return more than one result");
    return mlir::failure();
  }

  if (Parent->getNumResults() == 0) {
    if (Op->getNumOperands() != 0) {
      emitOpError("cannot return a result from an 'arc.if' without result");
      return mlir::failure();
    }
    return mlir::success();
  }

  if (Op->getOperand(0).getType() != Parent->getResult(0).getType()) {
    emitOpError("result type does not match the type of the parent: expected ")
        << Parent->getResult(0).getType() << " but found "
        << Op->getOperand(0).getType();
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult IfOp::customVerify() {
  if (this->getOperation()->getNumResults() > 1) {
    emitOpError("cannot return more than one result");
    return mlir::failure();
  }
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
// StateAppender operations
//===----------------------------------------------------------------------===//
LogicalResult StateAppenderFoldOp::customVerify() {
  auto InitTy = init().getType();
  auto ResultTy = res().getType();
  Operation *Callee =
      ::mlir::SymbolTable::lookupNearestSymbolFrom(this->getOperation(), fun());
  auto StateTy = state().getType().cast<ArconAppenderType>().getType();

  FuncOp F = dyn_cast<FuncOp>(Callee);
  FunctionType FT = F.type().dyn_cast<FunctionType>();

  if (!F)
    return emitOpError("fold function operand is not a function ");

  if (FT.getNumInputs() != 2)
    return emitOpError("folding function has the wrong number of operands, "
                       "expected 2, found ")
           << FT.getNumInputs();

  if (FT.getNumResults() != 1)
    return emitOpError("folding function has to return a single value, found ")
           << FT.getNumResults() << " values";

  if (InitTy != ResultTy)
    return emitOpError("expected init type ")
           << InitTy << " to match result type " << ResultTy;

  if (InitTy != FT.getInput(0))
    return emitOpError("expected type of accumulator initializer")
           << " to match type of folding function accumulator, found "
           << FT.getResult(0) << " expected " << InitTy;

  if (FT.getResult(0) != FT.getInput(0))
    return emitOpError("expected type of folding function accumulator to")
           << " match folding function result type, found " << FT.getResult(0)
           << " expected " << FT.getInput(0);

  if (StateTy != FT.getInput(1))
    return emitOpError("expected type of folding function input to")
           << " match appender type, found " << FT.getResult(0) << " expected "
           << FT.getInput(1);

  return mlir::success();
}

LogicalResult StateAppenderPushOp::customVerify() {
  auto ValTy = value().getType();
  auto StateTy = state().getType().cast<ArconAppenderType>().getType();
  if (ValTy != StateTy)
    return emitOpError("can't push a value of type ")
           << ValTy << " to an appender expecting type " << StateTy;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StateMap operations
//===----------------------------------------------------------------------===//
LogicalResult StateMapContainsOp::customVerify() {
  auto KeyTy = key().getType();
  auto ExpectedKeyTy = state().getType().cast<ArconMapType>().getKeyType();
  if (KeyTy != ExpectedKeyTy)
    return emitOpError("key type ")
           << KeyTy << " does not match map key type " << ExpectedKeyTy;
  return mlir::success();
}

LogicalResult StateMapGetOp::customVerify() {
  auto ValTy = result().getType();
  auto KeyTy = key().getType();
  auto ExpectedKeyTy = state().getType().cast<ArconMapType>().getKeyType();
  auto ExpectedValTy = state().getType().cast<ArconMapType>().getValueType();
  if (KeyTy != ExpectedKeyTy)
    return emitOpError("key type ")
           << KeyTy << " does not match map key type " << ExpectedKeyTy;
  if (ValTy != ExpectedValTy)
    return emitOpError("result type ")
           << ValTy << " does not match map value type " << ExpectedValTy;
  return mlir::success();
}

LogicalResult StateMapInsertOp::customVerify() {
  auto ValTy = value().getType();
  auto KeyTy = key().getType();
  auto ExpectedKeyTy = state().getType().cast<ArconMapType>().getKeyType();
  auto ExpectedValTy = state().getType().cast<ArconMapType>().getValueType();
  if (KeyTy != ExpectedKeyTy)
    return emitOpError("key type ")
           << KeyTy << " does not match map key type " << ExpectedKeyTy;
  if (ValTy != ExpectedValTy)
    return emitOpError("value type ")
           << ValTy << " does not match map value type " << ExpectedValTy;
  return mlir::success();
}

LogicalResult StateMapRemoveOp::customVerify() {
  auto KeyTy = key().getType();
  auto ExpectedKeyTy = state().getType().cast<ArconMapType>().getKeyType();
  if (KeyTy != ExpectedKeyTy)
    return emitOpError("key type ")
           << KeyTy << " does not match map key type " << ExpectedKeyTy;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// StateValue operations
//===----------------------------------------------------------------------===//
LogicalResult StateValueWriteOp::customVerify() {
  auto ValTy = value().getType();
  auto StateTy = state().getType().cast<ArconValueType>().getType();
  if (ValTy != StateTy)
    return emitOpError("Can't write a value of type ")
           << ValTy << " to a state value of type" << StateTy;
  return mlir::success();
}

LogicalResult StateValueReadOp::customVerify() {
  auto Operation = this->getOperation();
  auto ValTy = state().getType().cast<ArconValueType>().getType();
  auto ResultTy = Operation->getResult(0).getType();
  if (ValTy != ResultTy)
    return emitOpError("Expected result type ") << ValTy << " not " << ResultTy;
  return mlir::success();
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
// Arc types
//===----------------------------------------------------------------------===//
namespace arc {
namespace types {

//===----------------------------------------------------------------------===//
// Shared Functions
//===----------------------------------------------------------------------===//

bool isValueType(Type type) {
  if (type.isa<Float32Type>() || type.isa<Float64Type>() ||
      type.isa<IntegerType>() || type.isa<VectorType>() ||
      type.isa<RankedTensorType>() || type.isa<UnrankedTensorType>() ||
      type.isa<UnrankedMemRefType>() || type.isa<MemRefType>() ||
      type.isa<ComplexType>() || type.isa<ComplexType>() ||
      type.isa<NoneType>())
    return true;
  if (type.isa<TupleType>()) {
    for (auto t : type.cast<TupleType>().getTypes())
      if (!isValueType(t))
        return false;
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ArconType
//===----------------------------------------------------------------------===//

struct ArconTypeStorage : public TypeStorage {
  ArconTypeStorage(Type containedTy, std::string keyword)
      : TypeStorage(), containedType(containedTy), keyword(keyword) {}

  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == containedType; }

  Type containedType;
  std::string keyword;
};

Type ArconType::getContainedType() const {
  return static_cast<ImplType *>(impl)->containedType;
}

StringRef ArconType::getKeyword() const {
  return static_cast<ImplType *>(impl)->keyword;
}

bool isBuilderType(Type type) { return type.isa<AppenderType>(); }

void ArconType::print(DialectAsmPrinter &os) const {
  os << getKeyword() << "<" << getContainedType() << ">";
}

//===----------------------------------------------------------------------===//
// ArconAppenderType
//===----------------------------------------------------------------------===//
struct ArconAppenderTypeStorage : public ArconTypeStorage {
  using KeyTy = Type;

  ArconAppenderTypeStorage(Type elementType)
      : ArconTypeStorage(elementType, "arcon.appender") {}

  static ArconAppenderTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ArconValueTypeStorage>())
        ArconAppenderTypeStorage(key);
  }
};

ArconAppenderType ArconAppenderType::get(mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// Returns the element type of this stream type.
mlir::Type ArconAppenderType::getType() const { return getContainedType(); }

Type ArconAppenderType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;

  mlir::Type elementType;
  if (parser.parseType(elementType))
    return nullptr;

  if (parser.parseGreater())
    return Type();
  return ArconAppenderType::get(elementType);
}

//===----------------------------------------------------------------------===//
// ArconMapType
//===----------------------------------------------------------------------===//
struct ArconMapTypeStorage : public ArconTypeStorage {
  using KeyTy = std::pair<Type, Type>;

  ArconMapTypeStorage(Type keyType, Type valueType)
      : ArconTypeStorage(keyType, "arcon.map"), valueType(valueType) {}

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static KeyTy getKey(Type keyType, Type valueType) {
    return KeyTy(keyType, valueType);
  }

  bool operator==(const KeyTy &key) const {
    return key.first == containedType && key.second == valueType;
  }

  static ArconMapTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<ArconMapTypeStorage>())
        ArconMapTypeStorage(key.first, key.second);
  }

  Type valueType;
};

ArconMapType ArconMapType::get(mlir::Type keyType, mlir::Type valueType) {
  mlir::MLIRContext *ctx = keyType.getContext();
  return Base::get(ctx, keyType, valueType);
}

mlir::Type ArconMapType::getKeyType() const { return getContainedType(); }

mlir::Type ArconMapType::getValueType() const {

  return static_cast<ImplType *>(impl)->valueType;
}

Type ArconMapType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;

  mlir::Type keyType;
  if (parser.parseType(keyType))
    return nullptr;

  if (parser.parseComma())
    return nullptr;

  mlir::Type valueType;
  if (parser.parseType(valueType))
    return nullptr;

  if (parser.parseGreater())
    return Type();
  return ArconMapType::get(keyType, valueType);
}

void ArconMapType::print(DialectAsmPrinter &os) const {
  os << getKeyword() << "<" << getKeyType() << ", " << getValueType() << ">";
}

//===----------------------------------------------------------------------===//
// ArconValueType
//===----------------------------------------------------------------------===//
struct ArconValueTypeStorage : public ArconTypeStorage {
  using KeyTy = Type;

  ArconValueTypeStorage(Type elementType)
      : ArconTypeStorage(elementType, "arcon.value") {}

  static ArconValueTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<ArconValueTypeStorage>())
        ArconValueTypeStorage(key);
  }
};

ArconValueType ArconValueType::get(mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// Returns the element type of this stream type.
mlir::Type ArconValueType::getType() const { return getContainedType(); }

Type ArconValueType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;

  mlir::Type elementType;
  if (parser.parseType(elementType))
    return nullptr;

  if (parser.parseGreater())
    return Type();
  return ArconValueType::get(elementType);
}

//===----------------------------------------------------------------------===//
// BuilderType
//===----------------------------------------------------------------------===//

struct BuilderTypeStorage : public TypeStorage {
  BuilderTypeStorage(Type mergeTy, Type resultTy)
      : TypeStorage(), mergeType(mergeTy), resultType(resultTy) {}

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
  return Base::get(mergeType.getContext(), mergeType, resultType);
}

AppenderType
AppenderType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                         Type mergeType, RankedTensorType resultType,
                         Location loc) {
  return Base::getChecked(emitError, mergeType.getContext(), mergeType,
                          resultType);
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

  ArrayRef<int64_t> shape = {};
  RankedTensorType resultType = RankedTensorType::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc), shape, mergeType);

  return AppenderType::getChecked(mlir::detail::getDefaultDiagnosticEmitFn(loc),
                                  mergeType, resultType, loc);
}

void AppenderType::print(DialectAsmPrinter &os) const {
  os << "appender" << '<' << getMergeType() << '>';
}

LogicalResult AppenderType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   Type mergeType,
                                   RankedTensorType resultType) {
  if (!isValueType(mergeType)) {
    return emitError() << "appender merge type must be a value type: found "
                       << mergeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//
struct StreamTypeStorage : public ArconTypeStorage {
  using KeyTy = Type;

  StreamTypeStorage(Type elementType)
      : ArconTypeStorage(elementType, "stream") {}

  static StreamTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StreamTypeStorage>()) StreamTypeStorage(key);
  }
};

StreamType StreamType::get(mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

/// Returns the element type of this stream type.
mlir::Type StreamType::getType() const { return getContainedType(); }

Type StreamType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;

  mlir::Type elementType;
  if (parser.parseType(elementType))
    return nullptr;

  if (parser.parseGreater())
    return Type();
  return StreamType::get(elementType);
}

//===----------------------------------------------------------------------===//
// EnumType
//===----------------------------------------------------------------------===//
struct EnumTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<EnumType::VariantTy>;

  EnumTypeStorage(llvm::ArrayRef<EnumType::VariantTy> variantTypes)
      : variants(variantTypes) {}

  bool operator==(const KeyTy &key) const { return key == variants; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<EnumType::VariantTy> variantTypes) {
    return KeyTy(variantTypes);
  }

  static EnumTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    llvm::ArrayRef<EnumType::VariantTy> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<EnumTypeStorage>())
        EnumTypeStorage(elementTypes);
  }

  llvm::ArrayRef<EnumType::VariantTy> variants;
};

EnumType EnumType::get(llvm::ArrayRef<EnumType::VariantTy> variantTypes) {
  assert(!variantTypes.empty() && "expected at least 1 variant type");

  mlir::MLIRContext *ctx = variantTypes.front().second.getContext();
  return Base::get(ctx, variantTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<EnumType::VariantTy> EnumType::getVariants() const {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->variants;
}

size_t EnumType::getNumVariants() const { return getVariants().size(); }

Type EnumType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;
  Builder &builder = parser.getBuilder();

  SmallVector<EnumType::VariantTy, 3> variantTypes;
  do {
    StringRef name;
    if (parser.parseKeyword(&name) || parser.parseColon())
      return nullptr;

    EnumType::VariantTy variantType;
    variantType.first = StringAttr::get(builder.getContext(), name);
    if (parser.parseType(variantType.second))
      return nullptr;

    variantTypes.push_back(variantType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();
  return EnumType::get(variantTypes);
}

void EnumType::print(DialectAsmPrinter &os) const {
  // Print the struct type according to the parser format.
  os << "enum<";
  auto variants = getVariants();
  for (unsigned i = 0; i < getNumVariants(); i++) {
    if (i != 0)
      os << ", ";
    os << variants[i].first.getValue() << " : " << variants[i].second;
  }
  os << '>';
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

StructType StructType::get(mlir::MLIRContext *ctx,
                           llvm::ArrayRef<StructType::FieldTy> elementTypes) {
  return Base::get(ctx, elementTypes);
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
  while (true) {
    StringRef name;

    if (succeeded(parser.parseOptionalGreater()))
      return StructType::get(parser.getBuilder().getContext(), elementTypes);

    if (parser.parseKeyword(&name) || parser.parseColon())
      return nullptr;

    StructType::FieldTy elementType;
    elementType.first = StringAttr::get(builder.getContext(), name);
    if (parser.parseType(elementType.second))
      return nullptr;

    elementTypes.push_back(elementType);
    parser.parseOptionalComma();
  }
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

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Arc/Arc.cpp.inc"
