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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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
  addTypes<ADTType>();
  addTypes<ADTGenericType>();
  addTypes<SinkStreamType>();
  addTypes<SourceStreamType>();
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
  if (keyword == "adt")
    return ADTType::parse(parser);
  if (keyword == "generic_adt")
    return ADTGenericType::parse(parser);
  if (keyword == "stream.sink")
    return SinkStreamType::parse(parser);
  if (keyword == "stream.source")
    return SourceStreamType::parse(parser);
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
  if (auto t = type.dyn_cast<ADTType>())
    t.print(os);
  else if (auto t = type.dyn_cast<ADTGenericType>())
    t.print(os);
  else if (auto t = type.dyn_cast<SinkStreamType>())
    t.print(os);
  else if (auto t = type.dyn_cast<SourceStreamType>())
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
OpFoldResult arc::AndOp::fold(FoldAdaptor operands) {
  /// and(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  /// and(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  return constFoldBinaryOp<IntegerAttr>(operands.getOperands(),
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// ConstantIntOp
//===----------------------------------------------------------------------===//
ParseResult ConstantIntOp::parse(OpAsmParser &parser, OperationState &state) {
  Attribute value;
  if (parser.parseAttribute(value, "value", state.attributes))
    return failure();
  Type type = value.cast<TypedAttr>().getType();
  return parser.addTypeToList(type, state.types);
}

void arc::ConstantIntOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getValue();
}

LogicalResult arc::ConstantIntOp::verify() {
  auto opType = getType();
  TypedAttr v = getValue();
  auto valueType = v.getType();

  // ODS already generates checks to make sure the result type is
  // valid. We just need to additionally check that the value's
  // attribute type is consistent with the result type.
  if (v.isa<IntegerAttr>()) {
    if (valueType != opType)
      return emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  } else {
    return emitOpError("cannot have value of type ") << valueType;
  }

  return success();
}

OpFoldResult ConstantIntOp::fold(FoldAdaptor operands) {
  assert(operands.getOperands().empty() && "constant has no operands");
  return getValue();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type. Stolen from the standard dialect.
Operation *ArcDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {

  if (type.isSignedInteger() || type.isUnsignedInteger())
    return builder.create<ConstantIntOp>(loc, type, value);

  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, value);

  if (func::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<func::ConstantOp>(loc, type,
                                            value.cast<FlatSymbolRefAttr>());
  }

  return nullptr;
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
OpFoldResult arc::CmpIOp::fold(FoldAdaptor operands) {
  assert(operands.getOperands().size() == 2 && "cmpi takes two arguments");

  auto lhs = operands.getLhs().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.getRhs().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};
  bool isUnsigned = lhs.cast<TypedAttr>().getType().isUnsignedInteger();
  auto val = applyCmpPredicate(getPredicate(), isUnsigned, lhs.getValue(),
                               rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// DivIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from standard dialect

OpFoldResult DivIOp::fold(FoldAdaptor operands) {
  assert(operands.getOperands().size() == 2 &&
         "binary operation takes two operands");

  bool isUnsigned = getType().isUnsignedInteger();
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands.getOperands(), [&](APInt a, APInt b) {
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
//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//
LogicalResult EnumAccessOp::verify() {
  auto ResultTy = getResult().getType();
  auto SourceTy = getValue().getType().cast<EnumType>();
  auto VariantTys = SourceTy.getVariants();
  auto WantedVariant = getVariant();

  if (ResultTy.isa<NoneType>())
    return emitError(": accessing a ")
           << ResultTy << "-typed variant does not make sense";

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

LogicalResult EnumCheckOp::verify() {
  auto SourceTy = getValue().getType().cast<EnumType>();
  auto VariantTys = SourceTy.getVariants();
  auto WantedVariant = (*this)->getAttrOfType<StringAttr>("variant").getValue();

  // Check that the given type matches the specified variant.
  for (auto &i : VariantTys)
    if (i.first.getValue().equals(WantedVariant))
      return mlir::success();
  return emitOpError(": variant '")
         << WantedVariant << "' does not exist in " << SourceTy;
}

LogicalResult MakeEnumOp::verify() {
  auto ResultTy = getResult().getType().cast<EnumType>();
  Type SourceTy = NoneType::get(getContext());
  auto VariantTys = ResultTy.getVariants();
  auto WantedVariant = getVariant();
  auto values = getValues();
  if (values.size() > 1)
    return emitOpError(": only a single value expected");

  if (values.size())
    SourceTy = values[0].getType();

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

LogicalResult MakeStructOp::verify() {
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

LogicalResult ArcBlockResultOp::verify() {
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

  if (Parent->getNumResults() != Op->getNumOperands()) {
    emitOpError(
        "the number of values returned does not match parent: expected ")
        << Parent->getNumResults() << " but found " << Op->getNumOperands()
        << " values";
    return mlir::failure();
  }

  if (Op->getOperand(0).getType() != Parent->getResult(0).getType()) {
    emitOpError("result type does not match the type of the parent: expected ")
        << Parent->getResult(0).getType() << " but found "
        << Op->getOperand(0).getType();
    return mlir::failure();
  }
  return mlir::success();
}

LogicalResult IfOp::verify() {
  if (this->getOperation()->getNumResults() > 1) {
    emitOpError("cannot return more than one result");
    return mlir::failure();
  }

  // Check that the terminators are a arc.loop.break or a
  // arc.block.result.
  auto &thenTerm = getThenRegion().getBlocks().back().back();
  auto &elseTerm = getElseRegion().getBlocks().back().back();

  if ((isa<ArcBlockResultOp>(thenTerm) || isa<LoopBreakOp>(thenTerm) ||
       isa<ArcReturnOp>(thenTerm)) &&
      (isa<ArcBlockResultOp>(elseTerm) || isa<LoopBreakOp>(elseTerm) ||
       isa<ArcReturnOp>(elseTerm)))
    return success();
  return emitOpError(
      "expects terminators to be 'arc.loop.break', 'arc.return' or"
      "'arc.block.result' operations");
  return mlir::success();
}

LogicalResult LoopBreakOp::verify() {
  // HasParent<"scf::WhileOp"> in the .td apparently only looks at the
  // immediate parent and not all parents. Therefore we have to check
  // that we are inside a loop here.
  scf::WhileOp loopOp = (*this)->getParentOfType<scf::WhileOp>();
  if (!loopOp)
    return emitOpError("must be inside a scf.while region");

  // Now check that what we return matches the type of the parent
  unsigned noofResults = getNumOperands();
  unsigned noofParentResults = loopOp.getNumResults();

  if (noofResults != noofParentResults)
    emitOpError("returns ")
        << noofResults << " values parent expects " << noofParentResults;

  auto breakTypes = getOperandTypes();
  auto loopTypes = loopOp.getResultTypes();
  for (unsigned i = 0; i < noofResults; i++)
    if (breakTypes[i] != loopTypes[i])
      emitOpError(
          "type signature does not match signature of parent 'scf.while'");

  return success();
}

LogicalResult StructAccessOp::verify() {
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

OpFoldResult AddIOp::fold(FoldAdaptor operands) {
  /// addi(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  bool isUnsigned = getType().isUnsignedInteger();
  bool overflowDetected = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands.getOperands(), [&](APInt a, APInt b) {
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
OpFoldResult MulIOp::fold(FoldAdaptor operands) {
  /// muli(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  /// muli(x, 1) -> x
  if (matchPattern(getRhs(), m_One()))
    return getOperand(0);

  bool isUnsigned = getType().isUnsignedInteger();

  // Don't fold if it would overflow
  bool overflow = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands.getOperands(), [&](APInt a, APInt b) {
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

OpFoldResult arc::OrOp::fold(FoldAdaptor operands) {
  /// or(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// or(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  return constFoldBinaryOp<IntegerAttr>(operands.getOperands(),
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// ReceiveOp
//===----------------------------------------------------------------------===//
LogicalResult ReceiveOp::verify() {
  // Check that we're located inside a task
  mlir::func::FuncOp function =
      getOperation()->getParentOfType<mlir::func::FuncOp>();
  if (!function->hasAttr("arc.is_task")) {
    emitOpError("can only be used inside a task");
    return mlir::failure();
  }
  // Check that the stream's element type matches what we receive
  auto ElemTy = getValue().getType();
  SourceStreamType StreamTy = getSource().getType().cast<SourceStreamType>();
  if (ElemTy != StreamTy.getElementType())
    return emitOpError("Can't receive a value of type ")
           << ElemTy << " from a " << StreamTy << " stream";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RemIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from standard dialect
OpFoldResult RemIOp::fold(FoldAdaptor operands) {
  assert(operands.getOperands().size() == 2 &&
         "remi_unsigned takes two operands");

  auto rhs = operands.getRhs().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOne())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isZero())
    return {};

  auto lhs = operands.getLhs().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  bool isUnsigned = getType().isUnsignedInteger();
  return IntegerAttr::get(lhs.getType(), isUnsigned
                                             ? lhs.getValue().urem(rhsValue)
                                             : lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// ArcReturnOp
//===----------------------------------------------------------------------===//
LogicalResult ArcReturnOp::verify() {
  mlir::func::FuncOp function = (*this)->getParentOfType<mlir::func::FuncOp>();
  if (!function)
    return emitOpError("expects parent op builtin.func");

  FunctionType funType = function.getFunctionType().cast<FunctionType>();

  if (funType.getNumResults() == 0 && getReturnedValue())
    return emitOpError("cannot return a value from a void function");

  if (!getReturnedValue() && funType.getNumResults())
    return emitOpError("operation must return a ")
           << funType.getResult(0) << " value";

  if (!funType.getNumResults())
    return success();

  Type returnType = getOperand(0).getType();
  Type funReturnType = funType.getResult(0);

  if (funReturnType != returnType) {
    return emitOpError("result type does not match the type of the "
                       "function: expected ")
           << funReturnType << " but found " << returnType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ArcSpawnOp
//===----------------------------------------------------------------------===//

LogicalResult ArcSpawnOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // In large parts stolen from the func::CallOp.
  using namespace mlir::func;
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  if (!fn->hasAttr("arc.is_task"))
    return emitOpError("'callee' must be a task");

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != 0)
    return emitOpError("'callee' should not have a result");
  return success();
}

FunctionType ArcSpawnOp::getCalleeType() {
  SmallVector<Type, 1> resultTypes;
  return FunctionType::get(getContext(), getOperandTypes(), resultTypes);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Stolen from standard dialect
OpFoldResult arc::SelectOp::fold(FoldAdaptor operands) {
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
// SendOp
//===----------------------------------------------------------------------===//
LogicalResult SendOp::verify() {
  // Check that we're located inside a task
  mlir::func::FuncOp function =
      getOperation()->getParentOfType<mlir::func::FuncOp>();
  if (!function->hasAttr("arc.is_task")) {
    emitOpError("can only be used inside a task");
    return mlir::failure();
  }
  // Check that the stream's element type matches what we send
  auto ElemTy = getValue().getType();
  SinkStreamType StreamTy = getSink().getType().cast<SinkStreamType>();
  if (ElemTy != StreamTy.getElementType())
    return emitOpError("Can't send value of type ")
           << ElemTy << " on a " << StreamTy << " stream";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//
// Mostly stolen from the standard dialect
OpFoldResult SubIOp::fold(FoldAdaptor operands) {
  /// addi(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  bool isUnsigned = getType().isUnsignedInteger();
  bool overflowDetected = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands.getOperands(), [&](APInt a, APInt b) {
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
OpFoldResult arc::XOrOp::fold(FoldAdaptor operands) {
  /// xor(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// xor(x,x) -> 0
  if (getLhs() == getRhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands.getOperands(),
                                        [](APInt a, APInt b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// FilterOp
//===----------------------------------------------------------------------===//
LogicalResult FilterOp::verify() {
  // The type correctnes of the predicate is checked in
  // verifySymbolUses(), here is enough to check that the types of our
  // input and ouput streams match.
  if (getInput().getType() != getOutput().getType())
    return emitOpError("input and output streams should have the same types");

  return mlir::success();
}

LogicalResult FilterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // In large parts stolen from the func::CallOp.
  using namespace mlir::func;
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("predicate");
  if (!fnAttr)
    return emitOpError("requires a 'predicate' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  // Verify that the predicate returns a boolean.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumResults() != 1)
    return emitOpError("incorrect number of results for predicate");
  if (fnType.getNumInputs() < 1 || fnType.getNumInputs() > 2)
    return emitOpError("incorrect number of operands for predicate");
  IntegerType returnType = fnType.getResult(0).dyn_cast<IntegerType>();
  if (!returnType || returnType.getWidth() != 1)
    return emitOpError("predicate does not return a boolean");

  // Check that the predicate arguments match up, we must have at
  // least one operand, and it must match the type of the stream.
  if (fnType.getNumInputs() < 1)
    return emitOpError("incorrect number of operands for predicate");
  StreamType sst = getInput().getType().cast<StreamType>();
  if (fnType.getInput(0) != sst.getElementType())
    return emitOpError("predicate type mismatch: expected operand type ")
           << fnType.getInput(0) << ", but received" << sst.getElementType();

  // If the predicate requires a closure, check that the types match up
  auto thunkAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("predicate_env_thunk");
  if (fnType.getNumInputs() == 2) {
    // The predicate requires an environment
    if (!thunkAttr)
      return emitOpError("predicate expects environment but no "
                         "'predicate_env_thunk' attribute set");
    FuncOp thunk =
        symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, thunkAttr);
    if (!thunk)
      return emitOpError() << "'" << thunkAttr.getValue()
                           << "' does not reference a valid function";

    auto thunkType = thunk.getFunctionType();
    if (thunkType.getNumInputs() != 0)
      return emitOpError("environment thunk cannot take any arguments");
    if (thunkType.getNumResults() != 1)
      return emitOpError("environment thunk must return a value");
    if (thunkType.getResult(0) != fnType.getInput(1))
      return emitOpError("predicate environment type mismatch: expected "
                         "operand type ")
             << fnType.getInput(1) << ", but thunk returns "
             << thunkType.getResult(0);
  } else {
    if (thunkAttr)
      return emitOpError("map function does not expect an environment but "
                         "'predicate_env_thunk' attribute set");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KeybyOp
//===----------------------------------------------------------------------===//
LogicalResult KeybyOp::verify() {
  // The type correctnes of the predicate is checked in
  // verifySymbolUses(), here is enough to check that the types of our
  // input and ouput streams match.
  if (getInput().getType().cast<StreamType>().getElementType() !=
      getOutput().getType().cast<StreamType>().getElementType())
    return emitOpError(
        "input and output streams should have the same element type");

  return mlir::success();
}

LogicalResult KeybyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // In large parts stolen from the func::CallOp.
  using namespace mlir::func;
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("key_fun");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  // Verify that the keyby function has the right number of arguments
  // and results.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() < 1 || fnType.getNumInputs() > 2)
    return emitOpError("incorrect number of operands for keyby function");
  if (fnType.getNumResults() != 1)
    return emitOpError("incorrect number of results for keyby function");

  // Check that the keyby function argument type matches the input
  // stream element type.
  Type inputType = getInput().getType().cast<StreamType>().getElementType();
  if (fnType.getInput(0) != inputType)
    return emitOpError("keyby function type mismatch: input stream contains ")
           << inputType << " but keyby function expects " << fnType.getInput(0);

  // Check that the environment thunk has the right type
  auto thunkAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("key_fun_env_thunk");
  if (!thunkAttr && fnType.getNumInputs() > 1)
    return emitOpError("keyby function expects environment but no "
                       "'key_fun_env_thunk' attribute set");
  if (thunkAttr && fnType.getNumInputs() == 1)
    return emitOpError("keyby function does not expect an environment but "
                       "'key_fun_env_thunk' attribute set");
  if (thunkAttr) {
    FuncOp thunk =
        symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, thunkAttr);
    if (!thunk)
      return emitOpError() << "'" << thunkAttr.getValue()
                           << "' does not reference a valid function";
    auto thunkType = thunk.getFunctionType();
    if (thunkType.getNumInputs() != 0)
      return emitOpError("environment thunk cannot take any arguments");
    if (thunkType.getNumResults() != 1)
      return emitOpError("environment thunk must return a value");
    if (thunkType.getResult(0) != fnType.getInput(1))
      return emitOpError("keyby function environment type mismatch: expected "
                         "operand type ")
             << fnType.getInput(1) << ", but thunk returns "
             << thunkType.getResult(0);
  }
  // Check that the return type matches the output stream element type.
  Type outputType = getOutput().getType().cast<StreamType>().getKeyType();
  Type returnType = fnType.getResult(0);

  if (returnType != outputType)
    return emitOpError(
               "keyby function type mismatch: output stream key type is ")
           << outputType << " but keyby function returns " << returnType;

  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//
LogicalResult MapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // In large parts stolen from the func::CallOp.
  using namespace mlir::func;
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("map_fun");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  // Verify that the map function has the right number of arguments
  // and results.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() < 1 || fnType.getNumInputs() > 2)
    return emitOpError("incorrect number of operands for map function");
  if (fnType.getNumResults() != 1)
    return emitOpError("incorrect number of results for map function");

  // Check that the map function argument type matches the input
  // stream element type.
  Type inputType = getInput().getType().cast<StreamType>().getElementType();
  if (fnType.getInput(0) != inputType)
    return emitOpError("map function type mismatch: input stream contains ")
           << inputType << " but map function expects " << fnType.getInput(0);

  // Check that the environment thunk has the right type
  auto thunkAttr =
      (*this)->getAttrOfType<FlatSymbolRefAttr>("map_fun_env_thunk");
  if (!thunkAttr && fnType.getNumInputs() > 1)
    return emitOpError("map function expects environment but no "
                       "'map_fun_env_thunk' attribute set");
  if (thunkAttr && fnType.getNumInputs() == 1)
    return emitOpError("map function does not expect an environment but "
                       "'map_fun_env_thunk' attribute set");
  if (thunkAttr) {
    FuncOp thunk =
        symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, thunkAttr);
    if (!thunk)
      return emitOpError() << "'" << thunkAttr.getValue()
                           << "' does not reference a valid function";
    auto thunkType = thunk.getFunctionType();
    if (thunkType.getNumInputs() != 0)
      return emitOpError("environment thunk cannot take any arguments");
    if (thunkType.getNumResults() != 1)
      return emitOpError("environment thunk must return a value");
    if (thunkType.getResult(0) != fnType.getInput(1))
      return emitOpError("map function environment type mismatch: expected "
                         "operand type ")
             << fnType.getInput(1) << ", but thunk returns "
             << thunkType.getResult(0);
  }
  // Check that the return type matches the output stream element type.
  Type outputType = getOutput().getType().cast<StreamType>().getElementType();
  Type returnType = fnType.getResult(0);
  if (returnType != outputType)
    return emitOpError("map function type mismatch: output stream contains ")
           << outputType << " but map function returns " << returnType;

  return success();
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
  return false;
}

//===----------------------------------------------------------------------===//
// ADTType
//===----------------------------------------------------------------------===//
struct ADTTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::string;

  ADTTypeStorage(KeyTy rustTypeName) : rustType(rustTypeName) {}

  bool operator==(const KeyTy &key) const { return rustType == key; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(KeyTy name) { return KeyTy(name); }

  static ADTTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<ADTTypeStorage>()) ADTTypeStorage(key);
  }

  KeyTy rustType;
};

ADTType ADTType::get(mlir::MLIRContext *ctx, StringRef key) {
  return Base::get(ctx, key);
}

StringRef ADTType::getTypeName() const { return getImpl()->rustType; }

Type ADTType::parse(DialectAsmParser &parser) {
  if (parser.parseLess())
    return nullptr;
  std::string str;
  if (parser.parseString(&str))
    return nullptr;
  if (parser.parseGreater())
    return nullptr;
  return ADTType::get(parser.getBuilder().getContext(), str);
}

void ADTType::print(DialectAsmPrinter &os) const {
  os << "adt<\"" << getTypeName() << "\">";
}

//===----------------------------------------------------------------------===//
// ADTTemplateType
//===----------------------------------------------------------------------===//
struct ADTGenericTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<std::string, llvm::ArrayRef<mlir::Type>>;

  ADTGenericTypeStorage(StringRef name, ArrayRef<Type> parms)
      : name(name), parameters(parms.begin(), parms.end()) {}

  bool operator==(const KeyTy &k) const { return k == KeyTy(name, parameters); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(std::string name, llvm::ArrayRef<mlir::Type> parameters) {
    return KeyTy(name, parameters);
  }

  static ADTGenericTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<ADTGenericTypeStorage>())
        ADTGenericTypeStorage(key.first, key.second);
  }

  std::string name;
  SmallVector<Type, 4> parameters;
};

ADTGenericType ADTGenericType::get(mlir::MLIRContext *ctx, StringRef name,
                                   llvm::ArrayRef<mlir::Type> parameterTypes) {
  return Base::get(ctx, name, parameterTypes);
}

StringRef ADTGenericType::getTemplateName() const { return getImpl()->name; }

llvm::ArrayRef<mlir::Type> ADTGenericType::getParameterTypes() const {
  return getImpl()->parameters;
}

Type ADTGenericType::parse(DialectAsmParser &parser) {
  SmallVector<mlir::Type, 3> parameterTypes;

  if (parser.parseLess())
    return nullptr;
  std::string typeName;
  if (parser.parseString(&typeName))
    return nullptr;
  while (true) {
    if (succeeded(parser.parseOptionalGreater()))
      return ADTGenericType::get(parser.getBuilder().getContext(), typeName,
                                 parameterTypes);
    if (parser.parseComma())
      return nullptr;
    mlir::Type ty;
    if (parser.parseType(ty))
      return nullptr;
    parameterTypes.push_back(ty);
  }
}

void ADTGenericType::print(DialectAsmPrinter &os) const {
  os << "generic_adt<\"" << getTemplateName();
  for (mlir::Type t : getParameterTypes())
    os << ", " << t;
  os << "\">";
}

//===----------------------------------------------------------------------===//
// StreamTypeBase
//===----------------------------------------------------------------------===//

struct StreamTypeBaseStorage : public TypeStorage {
  StreamTypeBaseStorage(Type keyTy, Type containedTy, std::string keyword)
      : TypeStorage(), keyType(keyTy), containedType(containedTy),
        keyword(keyword) {}

  using KeyTy = std::pair<Type, Type>;

  bool operator==(const KeyTy &key) const {
    return key.first == keyType && key.second == containedType;
  }

  Type keyType;
  Type containedType;
  std::string keyword;
};

Type StreamTypeBase::getElementType() const {
  return static_cast<ImplType *>(impl)->containedType;
}

Type StreamTypeBase::getKeyType() const {
  return static_cast<ImplType *>(impl)->keyType;
}

StringRef StreamTypeBase::getKeyword() const {
  return static_cast<ImplType *>(impl)->keyword;
}

void StreamTypeBase::print(DialectAsmPrinter &os) const {
  os << getKeyword() << "<" << getKeyType() << ", " << getElementType() << ">";
}

Optional<std::pair<Type, Type>>
StreamTypeBase::parseStreamType(DialectAsmParser &parser) {
  using ST = std::pair<Type, Type>;
  if (parser.parseLess())
    return Optional<ST>();

  mlir::Type keyType;
  if (parser.parseType(keyType))
    return Optional<ST>();

  if (parser.parseComma())
    return Optional<ST>();

  mlir::Type elementType;
  if (parser.parseType(elementType))
    return Optional<ST>();

  if (parser.parseGreater())
    return Optional<ST>();
  return Optional<ST>(std::make_pair(keyType, elementType));
}

//===----------------------------------------------------------------------===//
// SinkStreamType
//===----------------------------------------------------------------------===//
struct SinkStreamTypeStorage : public StreamTypeBaseStorage {
  using KeyTy = std::pair<Type, Type>;

  SinkStreamTypeStorage(Type keyType, Type elementType)
      : StreamTypeBaseStorage(keyType, elementType, "stream.sink") {}

  static SinkStreamTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<SinkStreamTypeStorage>())
        SinkStreamTypeStorage(key.first, key.second);
  }
};

SinkStreamType SinkStreamType::get(mlir::Type keyType, mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, keyType, elementType);
}

Type SinkStreamType::parse(DialectAsmParser &parser) {
  Optional<std::pair<Type, Type>> t = parseStreamType(parser);
  if (t)
    return SinkStreamType::get(t->first, t->second);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SourceStreamType
//===----------------------------------------------------------------------===//
struct SourceStreamTypeStorage : public StreamTypeBaseStorage {
  using KeyTy = std::pair<Type, Type>;

  SourceStreamTypeStorage(Type keyType, Type elementType)
      : StreamTypeBaseStorage(keyType, elementType, "stream.source") {}

  static SourceStreamTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<SourceStreamTypeStorage>())
        SourceStreamTypeStorage(key.first, key.second);
  }
};

SourceStreamType SourceStreamType::get(mlir::Type keyType,
                                       mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, keyType, elementType);
}

Type SourceStreamType::parse(DialectAsmParser &parser) {
  auto t = parseStreamType(parser);
  if (t)
    return SourceStreamType::get(t->first, t->second);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//
struct StreamTypeStorage : public StreamTypeBaseStorage {
  using KeyTy = std::pair<Type, Type>;

  StreamTypeStorage(Type keyType, Type elementType)
      : StreamTypeBaseStorage(keyType, elementType, "stream") {}

  static StreamTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<SinkStreamTypeStorage>())
        StreamTypeStorage(key.first, key.second);
  }
};

StreamType StreamType::get(mlir::Type keyType, mlir::Type elementType) {
  mlir::MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, keyType, elementType);
}

Type StreamType::parse(DialectAsmParser &parser) {
  Optional<std::pair<Type, Type>> t = parseStreamType(parser);
  if (t)
    return StreamType::get(t->first, t->second);
  return nullptr;
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
  using KeyTy = std::pair<bool, llvm::ArrayRef<StructType::FieldTy>>;

  StructTypeStorage(bool isCompact,
                    llvm::ArrayRef<StructType::FieldTy> elementTypes)
      : isCompact(isCompact), fields(elementTypes) {}

  bool operator==(const KeyTy &key) const {
    return key.first == isCompact && key.second == fields;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(llvm::hash_value(key.first),
                              llvm::hash_value(key.second));
  }

  static KeyTy getKey(bool isCompact,
                      llvm::ArrayRef<StructType::FieldTy> elementTypes) {
    return KeyTy(isCompact, elementTypes);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<StructType::FieldTy> elementTypes =
        allocator.copyInto(key.second);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(key.first, elementTypes);
  }

  bool isCompact;
  llvm::ArrayRef<StructType::FieldTy> fields;
};

StructType StructType::get(mlir::MLIRContext *ctx, bool isCompact,
                           llvm::ArrayRef<StructType::FieldTy> elementTypes) {
  return Base::get(ctx, isCompact, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<StructType::FieldTy> StructType::getFields() const {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->fields;
}

size_t StructType::getNumFields() const { return getFields().size(); }

bool StructType::isCompact() const { return getImpl()->isCompact; }

Type StructType::parse(DialectAsmParser &parser) {
  bool isCompact = false;

  if (parser.parseLess())
    return nullptr;

  if (!parser.parseOptionalLess())
    isCompact = true;

  Builder &builder = parser.getBuilder();

  SmallVector<StructType::FieldTy, 3> elementTypes;
  while (true) {
    StringRef name;

    if (succeeded(parser.parseOptionalGreater())) {
      if (isCompact && parser.parseGreater())
        return nullptr;

      return StructType::get(parser.getBuilder().getContext(), isCompact,
                             elementTypes);
    }

    if (parser.parseKeyword(&name) || parser.parseColon())
      return nullptr;

    StructType::FieldTy elementType;
    elementType.first = StringAttr::get(builder.getContext(), name);
    if (parser.parseType(elementType.second))
      return nullptr;

    elementTypes.push_back(elementType);
    (void)parser.parseOptionalComma();
  }
}

void StructType::print(DialectAsmPrinter &os) const {
  // Print the struct type according to the parser format.
  os << "struct";
  if (isCompact())
    os << "<";
  os << "<";
  auto fields = getFields();
  for (unsigned i = 0; i < getNumFields(); i++) {
    if (i != 0)
      os << ", ";
    os << fields[i].first.getValue() << " : " << fields[i].second;
  }
  os << '>';
  if (isCompact())
    os << ">";
}

} // namespace types
} // namespace arc

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Arc/Arc.cpp.inc"
