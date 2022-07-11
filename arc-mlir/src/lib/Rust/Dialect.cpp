//===- Rust IR Dialect registration in MLIR -===//
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
// This file implements the dialect for the Rust IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "Rust/Rust.h"
#include "Rust/RustPrinterStream.h"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include "Rust/RustDialect.cpp.inc"

using namespace mlir;
using namespace rust;
using namespace types;

static llvm::cl::opt<std::string>
    crateNameOverride("rustcratename",
                      llvm::cl::desc("Override name of output crate"),
                      llvm::cl::value_desc("cratename"));

static llvm::cl::opt<std::string>
    rustModuleFile("rustfile",
                   llvm::cl::desc("Write all rust output to a single file"),
                   llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    rustInclude("rustinclude",
                llvm::cl::desc("Include this file into the generated module"),
                llvm::cl::value_desc("filename"));

//===----------------------------------------------------------------------===//
// RustDialect
//===----------------------------------------------------------------------===//

void RustDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Rust/Rust.cpp.inc"
      >();
  addTypes<RustType>();
  addTypes<RustEnumType>();
  addTypes<RustStructType>();
  addTypes<RustStreamType>();
  addTypes<RustSinkStreamType>();
  addTypes<RustSourceStreamType>();
  addTypes<RustTensorType>();
  addTypes<RustGenericADTType>();
  addTypes<RustTupleType>();

  auto ctx = getContext();

  floatTy = RustType::get(ctx, "f32");
  doubleTy = RustType::get(ctx, "f64");
  boolTy = RustType::get(ctx, "bool");
  i8Ty = RustType::get(ctx, "i8");
  i16Ty = RustType::get(ctx, "i16");
  i32Ty = RustType::get(ctx, "i32");
  i64Ty = RustType::get(ctx, "i64");
  u8Ty = RustType::get(ctx, "u8");
  u16Ty = RustType::get(ctx, "u16");
  u32Ty = RustType::get(ctx, "u32");
  u64Ty = RustType::get(ctx, "u64");
  noneTy = RustType::get(ctx, "unit");
}

//===----------------------------------------------------------------------===//
// RustDialect Type Parsing
//===----------------------------------------------------------------------===//

Type RustDialect::parseType(DialectAsmParser &parser) const {
  //  StringRef type;
  StringRef tyData = parser.getFullSymbolSpec();
  // if (failed(parser.parseKeyword(&type)))
  //   return nullptr;
  return RustType::get(getContext(), tyData);
}

//===----------------------------------------------------------------------===//
// RustDialect Type Printing
//===----------------------------------------------------------------------===//

static void printAsMLIR(Type type, DialectAsmPrinter &os) {
  if (auto t = type.dyn_cast<RustType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustStreamType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustSinkStreamType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustSourceStreamType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustStructType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustGenericADTType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustTensorType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustTupleType>())
    t.printAsMLIR(os);
  else if (auto t = type.dyn_cast<RustEnumType>())
    t.printAsMLIR(os);
  else
    llvm_unreachable("Unhandled Rust type");
}

void RustDialect::printType(Type type, DialectAsmPrinter &os) const {
  ::printAsMLIR(type, os);
}

//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//

/// Hook for FunctionLike verifier.
LogicalResult RustFuncOp::verifyType() {
  Type type = getFunctionType();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

/// Hook for FunctionLike verifier.
LogicalResult RustExtFuncOp::verifyType() {
  Type type = getFunctionType();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

/// Verifies the body of the function.
LogicalResult RustFuncOp::verifyBody() {
  unsigned numFuncArguments = getNumArguments();
  unsigned numBlockArguments = empty() ? 0 : front().getNumArguments();
  if (numBlockArguments != numFuncArguments)
    return emitOpError() << "expected " << numFuncArguments
                         << " arguments to body region, found "
                         << numBlockArguments;

  ArrayRef<Type> funcArgTypes = getArgumentTypes();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    Type blockArgType = front().getArgument(i).getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", found "
                           << blockArgType;
  }

  return success();
}

LogicalResult RustReturnOp::verify() {
  RustFuncOp function = (*this)->getParentOfType<RustFuncOp>();

  if (!function)
    return emitOpError("expects 'rust.func' parent");

  FunctionType funType = function.getFunctionType();

  if (funType.getNumResults() == 0 && operands())
    return emitOpError("cannot return a value from a void function");

  if (!operands() && funType.getNumResults())
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
// RustDialect Rust Printing
//===----------------------------------------------------------------------===//

static std::string getMangledName(Type type, rust::RustPrinterStream &ps) {
  if (auto t = type.dyn_cast<RustType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustStructType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustStreamType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustSinkStreamType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustSourceStreamType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustGenericADTType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustTensorType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustTupleType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<RustEnumType>())
    return t.getMangledName(ps);
  else if (auto t = type.dyn_cast<FunctionType>())
    return ps.getMangledName(t);
  return "<not-a-rust-type(mangled)>";
}

static void printAsRust(Type type, llvm::raw_ostream &o,
                        rust::RustPrinterStream &ps) {
  if (auto t = type.dyn_cast<RustType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustStructType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustStreamType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustSinkStreamType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustSourceStreamType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustGenericADTType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustTensorType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustTupleType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<RustEnumType>())
    t.printAsRust(o, ps);
  else if (auto t = type.dyn_cast<FunctionType>())
    ps.printAsRust(o, t);
  else
    ps << "<not-a-rust-type(rust)>";
}

LogicalResult rust::writeModuleAsInline(ModuleOp module, llvm::raw_ostream &o) {

  if (!module.getName()) {
    emitError(module.getLoc())
        << "Rust module is missing a name (is the module implicitly created?)";
    return failure();
  }

  RustPrinterStream PS(module.getName()->str(), rustInclude);

  for (Operation &operation : module) {
    if (RustFuncOp op = dyn_cast<RustFuncOp>(operation))
      op.writeRust(PS);
    else if (RustExtFuncOp op = dyn_cast<RustExtFuncOp>(operation))
      op.writeRust(PS);
  }

  PS.flush(o);

  return success();
}

static RustPrinterStream &writeRust(Operation &operation,
                                    RustPrinterStream &PS) {
  if (RustReturnOp op = dyn_cast<RustReturnOp>(operation))
    op.writeRust(PS);
  else if (RustConstantOp op = dyn_cast<RustConstantOp>(operation))
    op.writeRust(PS);
  else if (RustUnaryOp op = dyn_cast<RustUnaryOp>(operation))
    op.writeRust(PS);
  else if (RustBinaryOp op = dyn_cast<RustBinaryOp>(operation))
    op.writeRust(PS);
  else if (RustBinaryRcOp op = dyn_cast<RustBinaryRcOp>(operation))
    op.writeRust(PS);
  else if (RustCallOp op = dyn_cast<RustCallOp>(operation))
    op.writeRust(PS);
  else if (RustCallIndirectOp op = dyn_cast<RustCallIndirectOp>(operation))
    op.writeRust(PS);
  else if (RustCompOp op = dyn_cast<RustCompOp>(operation))
    op.writeRust(PS);
  else if (RustEnumAccessOp op = dyn_cast<RustEnumAccessOp>(operation))
    op.writeRust(PS);
  else if (RustEnumCheckOp op = dyn_cast<RustEnumCheckOp>(operation))
    op.writeRust(PS);
  else if (RustFieldAccessOp op = dyn_cast<RustFieldAccessOp>(operation))
    op.writeRust(PS);
  else if (RustIfOp op = dyn_cast<RustIfOp>(operation))
    op.writeRust(PS);
  else if (RustBlockResultOp op = dyn_cast<RustBlockResultOp>(operation))
    op.writeRust(PS);
  else if (RustLoopOp op = dyn_cast<RustLoopOp>(operation))
    op.writeRust(PS);
  else if (RustLoopBreakOp op = dyn_cast<RustLoopBreakOp>(operation))
    op.writeRust(PS);
  else if (RustLoopConditionOp op = dyn_cast<RustLoopConditionOp>(operation))
    op.writeRust(PS);
  else if (RustLoopYieldOp op = dyn_cast<RustLoopYieldOp>(operation))
    op.writeRust(PS);
  else if (RustMakeEnumOp op = dyn_cast<RustMakeEnumOp>(operation))
    op.writeRust(PS);
  else if (RustMakeStructOp op = dyn_cast<RustMakeStructOp>(operation))
    op.writeRust(PS);
  else if (RustMethodCallOp op = dyn_cast<RustMethodCallOp>(operation))
    op.writeRust(PS);
  else if (RustTensorOp op = dyn_cast<RustTensorOp>(operation))
    op.writeRust(PS);
  else if (RustTupleOp op = dyn_cast<RustTupleOp>(operation))
    op.writeRust(PS);
  else if (RustEmitOp op = dyn_cast<RustEmitOp>(operation))
    op.writeRust(PS);
  else if (RustPanicOp op = dyn_cast<RustPanicOp>(operation))
    op.writeRust(PS);
  else if (RustReceiveOp op = dyn_cast<RustReceiveOp>(operation))
    op.writeRust(PS);
  else if (RustSendOp op = dyn_cast<RustSendOp>(operation))
    op.writeRust(PS);
  else if (RustSpawnOp op = dyn_cast<RustSpawnOp>(operation))
    op.writeRust(PS);
  else {
    operation.emitError("Unsupported operation");
  }
  return PS;
}

std::string RustPrinterStream::getConstant(RustConstantOp v) {
  StringAttr str = v.getValue().dyn_cast<StringAttr>();
  if (FunctionType fType = v.getType().dyn_cast<FunctionType>()) {
    // Although a function reference is a constant in MLIR it is not
    // in our Rust dialect, so we need to handle them specially.

    if (Operation *target = SymbolTable::lookupNearestSymbolFrom(v, str)) {
      if (target->hasAttr("arc.rust_name"))
        str = target->getAttrOfType<StringAttr>("arc.rust_name");
      if (target->hasAttr("rust.declare"))
        DeclaredFunctions.insert(target);
    }

    auto found = Value2ID.find(v);
    int id = 0;
    if (found == Value2ID.end()) {
      id = NextID++;
      Value2ID[v] = id;
      Body << "let v" << id << " : ";

      Body << ::getMangledName(fType, *this);
      Body << " = function!(" << str.getValue() << ");\n";
    } else
      id = found->second;
    return "v" + std::to_string(id);
  }
  auto found = Value2ID.find(v);
  int id = 0;
  if (found == Value2ID.end()) {
    id = --NextConstID;
    Value2ID[v] = id;
  } else
    id = found->second;
  Constants << "const C" << -id << " : ";
  ::printAsRust(v.getType(), Constants, *this);
  Constants << " = " << str.getValue() << ";\n";
  return "C" + std::to_string(-id);
}

void RustCallOp::writeRust(RustPrinterStream &PS) {
  bool has_result = getNumResults();
  if (has_result) {
    auto r = getResult(0);
    PS << "let ";
    PS.printAsArg(r) << ":" << r.getType() << " = ";
  }

  StringRef callee = getCallee();
  StringAttr calleeName = StringAttr::get(this->getContext(), getCallee());
  Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, calleeName);
  if (target && target->hasAttr("arc.rust_name"))
    callee = target->getAttrOfType<StringAttr>("arc.rust_name").getValue();

  if (target && target->hasAttr("rust.async"))
    PS << "call_async!(";
  else
    PS << "call!(";
  PS << callee << "(";
  for (auto a : getOperands())
    PS << a << ", ";
  PS << "));\n";
}

void RustCallIndirectOp::writeRust(RustPrinterStream &PS) {
  bool has_result = getNumResults();
  if (has_result) {
    auto r = getResult(0);
    PS << "let ";
    PS.printAsArg(r) << ":" << r.getType() << " = ";
  }
  PS << "call_indirect!((" << getCallee() << ")(";
  for (auto a : getArgOperands())
    PS << a << ", ";
  PS << "));\n";
}

// Write this function as Rust code to os
void RustFuncOp::writeRust(RustPrinterStream &PS) {
  if ((*this)->hasAttr("arc.is_task"))
    PS.addTask(this);

  if ((*this)->hasAttr("rust.annotation"))
    PS << (*this)->getAttrOfType<StringAttr>("rust.annotation").getValue()
       << "\n";
  else
    PS << "#[rewrite]\n";
  PS << "pub ";
  if ((*this)->hasAttr("rust.async"))
    PS << "async ";
  PS << "fn ";
  if ((*this)->hasAttr("arc.rust_name"))
    PS << (*this)->getAttrOfType<StringAttr>("arc.rust_name").getValue();
  else
    PS << getName();
  PS << "(";

  // Dump the function arguments
  unsigned numFuncArguments = getNumArguments();
  for (unsigned i = 0; i < numFuncArguments; i++) {
    Value v = front().getArgument(i);
    Type t = v.getType();
    if (i != 0)
      PS << ", ";
    if ((*this)->hasAttr("arc.is_task")) {
      if (RustSinkStreamType st = t.dyn_cast<RustSinkStreamType>())
        PS << "#[output]";
    }
    PS.printAsArg(v) << ": " << v.getType();
  }
  PS << ") ";
  FunctionType funcTy = getFunctionType();
  if (funcTy.getNumResults()) { // The return type
    PS << "-> " << funcTy.getResult(0) << " ";
  }

  // Dumping the body
  PS << "{\n";
  for (Operation &operation : this->body().front()) {
    ::writeRust(operation, PS);
  }
  PS << "}\n";
  PS.clearAliases();
}

// Write this function as Rust code to os
void RustExtFuncOp::writeRust(RustPrinterStream &PS) {
  // External functions are normally declared elsewhere, so there is
  // no need for a "protype" to be output. A special case is where an
  // external function has the `rust.annotation` attribute in which
  // case it should be emitted as a function with an empty body and
  // with the annotation.

  if (!(*this)->hasAttr("rust.annotation"))
    return;

  PS << (*this)->getAttrOfType<StringAttr>("rust.annotation").getValue()
     << "\npub ";
  if ((*this)->hasAttr("rust.async"))
    PS << "async ";
  PS << "fn " << getName() << "(";

  FunctionType fType = getFunctionType();

  unsigned numFuncArguments = fType.getNumInputs();
  for (unsigned i = 0; i < numFuncArguments; i++) {
    Type t = fType.getInput(i);
    if (i != 0)
      PS << ", ";
    PS << "arg" << i << " : " << t;
  }
  PS << ") ";
  if (fType.getNumResults()) { // The return type
    PS << "-> " << fType.getResult(0) << " ";
  }
  PS << "{}\n";
}

void RustReturnOp::writeRust(RustPrinterStream &PS) {
  if (getNumOperands())
    PS << "return " << getOperand(0) << ";\n";
  else
    PS << "return;\n";
}

void RustConstantOp::writeRust(RustPrinterStream &PS) { PS.getConstant(*this); }

void RustUnaryOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = " << getOperator() << "("
                   << getOperand() << ")"
                   << ";\n";
}

void RustLoopOp::writeRust(RustPrinterStream &PS) {
  PS << "// Loop variables\n";
  Block::BlockArgListType before_args = before().front().getArguments();
  Block::BlockArgListType after_args = after().front().getArguments();
  OperandRange inits = this->inits();

  assert(inits.size() == before_args.size());

  // Construct a mutable variable for each loop variable
  for (unsigned idx = 0; idx < inits.size(); idx++) {
    Value v = before_args[idx];
    Value i = inits[idx];

    PS << "let mut ";
    PS.printAsArg(v) << ":" << v.getType() << " = ";
    PS.print(i) << ";\n";
  }

  // Construct variables for catching the result
  if (getNumResults() != 0)
    PS << "let (";
  for (unsigned i = 0; i < getNumResults(); i++) {
    auto r = getResult(i);
    PS.printAsArg(r) << ",";
  }
  if (getNumResults() != 0)
    PS << ") : (";
  for (unsigned i = 0; i < getNumResults(); i++) {
    auto r = getResult(i);
    PS << r.getType() << ",";
  }
  if (getNumResults() != 0)
    PS << ") = ";

  // Emit the loop body
  PS << "loop {\n";
  PS << "// Before\n";
  for (Operation &operation : before().front())
    ::writeRust(operation, PS);
  RustLoopConditionOp cond = getConditionOp();
  auto passed_on = cond.args();
  PS << "// Pass on state from the before to the after part\n";
  assert(passed_on.size() == after_args.size());

  for (unsigned idx = 0; idx < after_args.size(); idx++) {
    Value v = after_args[idx];
    Value i = passed_on[idx];

    PS << "let ";
    PS.printAsArg(v) << ":" << v.getType() << " = ";
    PS.print(i) << ";\n";
  }

  PS << "// After\n";
  for (Operation &operation : after().front())
    ::writeRust(operation, PS);

  RustLoopYieldOp yield = getYieldOp();
  auto updated = yield.results();
  PS << "// Update the loop variables for the next iteration\n";
  assert(before_args.size() == updated.size());
  for (unsigned idx = 0; idx < before_args.size(); idx++) {
    Value v = before_args[idx];
    Value u = updated[idx];

    PS.printAsLValue(v) << " = ";
    PS.print(u) << ";\n";
  }

  PS << "};\n";
}

LogicalResult RustLoopBreakOp::verify() {
  // HasParent<"RustLoopOp"> in the .td apparently only looks at the
  // immediate parent and not all parents. Therefore we have to check
  // that we are inside a loop here.
  RustLoopOp loopOp = (*this)->getParentOfType<RustLoopOp>();
  if (!loopOp)
    return emitOpError("must be inside a rust.loop region");

  // Now check that what we return matches the type of the parent
  unsigned noofResults = getNumOperands();
  unsigned noofParentResults = loopOp.getNumResults();

  if (noofResults != noofParentResults)
    return emitOpError("returns ")
           << noofResults << " values parent expects " << noofParentResults;

  auto breakTypes = getOperandTypes();
  auto loopTypes = loopOp.getResultTypes();
  for (unsigned i = 0; i < noofResults; i++)
    if (breakTypes[i] != loopTypes[i])
      return emitOpError(
          "type signature does not match signature of parent 'rust.loop'");

  return success();
}

void RustLoopBreakOp::writeRust(RustPrinterStream &PS) {
  PS << "break";
  if (getNumOperands() != 1) {
    PS << " (";
    for (auto arg : results())
      PS << arg << ",";
    PS << ")";
  }
  PS << ";\n";
}

void RustLoopConditionOp::writeRust(RustPrinterStream &PS) {
  PS << "// Loop condition\n";
  PS << "if !" << getOperand(0) << " {\n";
  PS << "break";
  if (getNumOperands() != 1) {
    PS << " (";
    for (auto arg : args())
      PS << arg << ",";
    PS << ")";
  }
  PS << ";\n";
  PS << "}\n";
}

void RustLoopYieldOp::writeRust(RustPrinterStream &PS) {
  PS << "// Loop yield\n";
}

void RustMakeEnumOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  RustEnumType et = r.getType().cast<RustEnumType>();
  PS << "let ";
  PS.printAsArg(r) << ":" << et << " = enwrap!(" << et << "::" << variant()
                   << ", ";
  if (values().size())
    PS << values()[0];
  else
    PS << "()";
  PS << ");\n";
}

void RustMakeStructOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  RustStructType st = r.getType().cast<RustStructType>();
  PS << "let ";
  PS.printAsArg(r) << ":" << st << " = new!(" << st << " { ";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    if (i != 0)
      PS << ", ";
    auto v = args[i];
    PS << st.getFieldName(i) << " : " << v;
  }
  PS << "});\n";
}

void RustMethodCallOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = " << obj() << "." << getMethod()
                   << "(";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    if (i != 0)
      PS << ", ";
    auto v = args[i];
    PS << v;
  }
  PS << ");\n";
}

void RustBinaryOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = " << LHS() << " "
                   << getOperator() << " " << RHS() << ";\n";
}

void RustBinaryRcOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = Rc::new(&*" << LHS() << " "
                   << getOperator() << " &*" << RHS() << ");\n";
}

void RustCompOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = " << LHS() << " "
                   << getOperator() << " " << RHS() << ";\n";
}

void RustEmitOp::writeRust(RustPrinterStream &PS) {
  PS << "self.emit(" << value() << ");\n";
}

void RustEnumAccessOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  RustEnumType et = theEnum().getType().cast<RustEnumType>();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = unwrap!(" << et
                   << "::" << getVariant() << ", " << theEnum() << ");\n";
}

void RustEnumCheckOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  RustEnumType et = theEnum().getType().cast<RustEnumType>();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = is!(" << et
                   << "::" << getVariant() << ", " << theEnum() << ");\n";
}

void RustFieldAccessOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = access!(" << aggregate() << ", "
                   << getField() << ");\n";
}

void RustIfOp::writeRust(RustPrinterStream &PS) {
  if (getNumResults() != 0) {
    auto r = getResult(0);
    PS << "let ";
    PS.printAsArg(r) << ":" << r.getType() << " =";
  }
  // No clone is needed here as it will be inserted by the block
  // result.
  PS << " if " << getOperand() << " {\n";
  for (Operation &operation : thenRegion().front())
    ::writeRust(operation, PS);
  PS << "} else {\n";
  for (Operation &operation : elseRegion().front())
    ::writeRust(operation, PS);
  PS << "};\n";
}

LogicalResult RustIfOp::verify() {
  // Check that the terminators are a rust.loop.break or a
  // rust.block.result.
  auto &thenTerm = thenRegion().getBlocks().back().back();
  auto &elseTerm = elseRegion().getBlocks().back().back();

  if ((isa<RustBlockResultOp>(thenTerm) || isa<RustLoopBreakOp>(thenTerm) ||
       isa<RustReturnOp>(thenTerm)) &&
      (isa<RustBlockResultOp>(elseTerm) || isa<RustLoopBreakOp>(elseTerm) ||
       isa<RustReturnOp>(elseTerm)))
    return success();
  return emitOpError("expects terminators to be 'rust.loop.break' or "
                     "'rust.block.result' operations");
}

//===----------------------------------------------------------------------===//
// RustLoopOp, stolen from SCF
//===----------------------------------------------------------------------===//

RustLoopConditionOp RustLoopOp::getConditionOp() {
  return cast<RustLoopConditionOp>(before().front().getTerminator());
}

RustLoopYieldOp RustLoopOp::getYieldOp() {
  return cast<RustLoopYieldOp>(after().front().getTerminator());
}

void RustBlockResultOp::writeRust(RustPrinterStream &PS) {
  if (getNumOperands() == 0) {
    PS << "// No value\n";
    return;
  }
  auto r = getOperand(0);
  PS << r << "\n";
}

void RustPanicOp::writeRust(RustPrinterStream &PS) {
  PS << "panic!(";
  if (msg().hasValue())
    PS << "\"" << msg().getValue() << "\"";
  PS << ");\n";
}

void RustReceiveOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = pull!(" << source() << ");\n";
}

void RustSendOp::writeRust(RustPrinterStream &PS) {
  PS << "push!(" << value() << "," << sink() << ");\n";
}

void RustSpawnOp::writeRust(RustPrinterStream &PS) {
  StringRef callee = getCallee();
  StringAttr calleeName = StringAttr::get(this->getContext(), getCallee());
  Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, calleeName);
  if (target && target->hasAttr("arc.rust_name"))
    callee = target->getAttrOfType<StringAttr>("arc.rust_name").getValue();

  PS << "spawn!(" << callee << ", ";
  for (auto a : getArgOperands())
    PS << a << ", ";
  PS << ");\n";
}

void RustTensorOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType()
                   << " = Rc::new(Array::from_shape_vec((";
  RustTensorType t = result().getType().cast<RustTensorType>();
  for (int64_t d : t.getDimensions())
    PS << d << ", ";
  PS << "), vec![";
  auto args = values();
  for (unsigned i = 0; i < args.size(); i++) {
    auto v = args[i];
    PS << v << ", ";
  }
  PS << "]).unwrap());\n";
}

void RustTupleOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let ";
  PS.printAsArg(r) << ":" << r.getType() << " = (";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    auto v = args[i];
    PS << v << ", ";
  }
  PS << ");\n";
}

//===----------------------------------------------------------------------===//
// Crate versions
//===----------------------------------------------------------------------===//
namespace rust {
const char *CrateVersions::ndarray = "0.13.0";
} // namespace rust

//===----------------------------------------------------------------------===//
// Rust types
//===----------------------------------------------------------------------===//
namespace rust {
namespace types {

struct RustTypeStorage : public TypeStorage {
  RustTypeStorage(std::string type) : rustType(type), mangledName(type) {
    if (rustType[0] == '"')
      mangledName = rustType.substr(1, rustType.length() - 2);
  }

  std::string rustType;
  std::string mangledName;

  using KeyTy = std::string;

  bool operator==(const KeyTy &key) const { return key == KeyTy(rustType); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<RustTypeStorage>()) RustTypeStorage(key);
  }

  void printAsMLIR(DialectAsmPrinter &os) const;
  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);

  std::string getMangledName(rust::RustPrinterStream &ps) const {
    return mangledName;
  };

  bool isBool() const;
};

RustType RustType::get(MLIRContext *context, StringRef type) {
  return Base::get(context, type);
}

std::string RustType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

void RustTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << rustType;
}

void RustType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustTypeStorage::printAsRust(llvm::raw_ostream &o,
                                  rust::RustPrinterStream &ps) {
  o << getMangledName(ps);
}

void RustType::printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

bool RustTypeStorage::isBool() const {
  return mangledName.compare("bool") == 0;
}

bool RustType::isBool() const { return getImpl()->isBool(); }

// bool RustType::isUnit() const { return getRustType().equals("()"); }

RustType RustType::getFloatTy(RustDialect *dialect) { return dialect->floatTy; }

RustType RustType::getDoubleTy(RustDialect *dialect) {
  return dialect->doubleTy;
}

RustType RustType::getNoneTy(RustDialect *dialect) { return dialect->noneTy; }

RustType RustType::getIntegerTy(RustDialect *dialect, IntegerType ty) {
  switch (ty.getWidth()) {
  case 1:
    return dialect->boolTy;
  case 8:
    return ty.isUnsigned() ? dialect->u8Ty : dialect->i8Ty;
  case 16:
    return ty.isUnsigned() ? dialect->u16Ty : dialect->i16Ty;
  case 32:
    return ty.isUnsigned() ? dialect->u32Ty : dialect->i32Ty;
  case 64:
    return ty.isUnsigned() ? dialect->u64Ty : dialect->i64Ty;
  default:
    return emitError(UnknownLoc::get(dialect->getContext()), "unhandled type"),
           nullptr;
  }
}

//===----------------------------------------------------------------------===//
// RustEnumType
//===----------------------------------------------------------------------===//

struct RustEnumTypeStorage : public TypeStorage {
  RustEnumTypeStorage(ArrayRef<RustEnumType::EnumVariantTy> fields)
      : enumVariants(fields.begin(), fields.end()) {}
  SmallVector<RustEnumType::EnumVariantTy, 4> enumVariants;

  using KeyTy = ArrayRef<RustEnumType::EnumVariantTy>;

  bool operator==(const KeyTy &key) const { return key == KeyTy(enumVariants); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustEnumTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<RustEnumTypeStorage>())
        RustEnumTypeStorage(key);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);
  void printAsMLIR(DialectAsmPrinter &os) const;

  StringRef getVariantName(unsigned idx) const;
  Type getVariantType(unsigned idx) const;
  unsigned getNumVariants() const;

  void emitNestedTypedefs(rust::RustPrinterStream &os) const;
  std::string getSignature() const;
  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;
};

RustEnumType RustEnumType::get(RustDialect *dialect,
                               ArrayRef<EnumVariantTy> fields) {
  return Base::get(dialect->getContext(), fields);
}

void RustEnumTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "enum<";
  for (unsigned i = 0; i < enumVariants.size(); i++) {
    if (i != 0)
      os << ", ";
    os << enumVariants[i].first.getValue();
    os << " : ";
    ::printAsMLIR(enumVariants[i].second, os);
  }
  os << ">";
}

void RustEnumType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustEnumType::printAsRust(llvm::raw_ostream &o,
                               rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

StringRef RustEnumType::getVariantName(unsigned idx) const {
  return getImpl()->getVariantName(idx);
}

StringRef RustEnumTypeStorage::getVariantName(unsigned idx) const {
  return enumVariants[idx].first.getValue();
}

Type RustEnumType::getVariantType(unsigned idx) const {
  return getImpl()->getVariantType(idx);
}

Type RustEnumTypeStorage::getVariantType(unsigned idx) const {
  return enumVariants[idx].second;
}

unsigned RustEnumType::getNumVariants() const {
  return getImpl()->getNumVariants();
}

unsigned RustEnumTypeStorage::getNumVariants() const {
  return enumVariants.size();
}

std::string RustEnumType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string RustEnumTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);
  mangled << "Enum";

  for (auto &f : enumVariants) {
    StringRef fieldName = f.first.getValue();
    mangled << fieldName.size() << fieldName;
    mangled << ::getMangledName(f.second, ps);
  }
  mangled << "End";
  mangledName = mangled.str();

  llvm::raw_ostream &tyStream = ps.getNamedTypesStream();
  tyStream << "#[rewrite]\n";
  tyStream << "pub enum " << mangledName << " {\n";
  for (unsigned i = 0; i < enumVariants.size(); i++) {
    tyStream << "  " << enumVariants[i].first.getValue() << "("
             << ::getMangledName(enumVariants[i].second, ps) << "),\n";
  }
  tyStream << "\n}\n";

  return mangledName;
}

void RustEnumTypeStorage::printAsRust(llvm::raw_ostream &o,
                                      rust::RustPrinterStream &ps) {
  o << getMangledName(ps);
}

//===----------------------------------------------------------------------===//
// RustStreamType
//===----------------------------------------------------------------------===//

struct RustStreamTypeStorage : public TypeStorage {
  RustStreamTypeStorage(Type item) : item(item) {}

  Type item;

  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == KeyTy(item); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustStreamTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<RustStreamTypeStorage>())
        RustStreamTypeStorage(key);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);
  void printAsMLIR(DialectAsmPrinter &os);

  Type getType() const;

  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;

  std::string getMangledName();
};

std::string RustStreamType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string RustStreamTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string mn = ::getMangledName(item, ps);

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);
  mangled << "Stream";
  mangled << mn.size() << mn << "End";
  mangledName = mangled.str();
  return mangledName;
}

RustStreamType RustStreamType::get(RustDialect *dialect, Type item) {
  return Base::get(dialect->getContext(), item);
}

void RustStreamTypeStorage::printAsMLIR(DialectAsmPrinter &os) {
  os << "stream<";
  ::printAsMLIR(item, os);
  os << ">";
}

void RustStreamType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustStreamType::printAsRust(llvm::raw_ostream &o,
                                 rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

void RustStreamTypeStorage::printAsRust(llvm::raw_ostream &o,
                                        rust::RustPrinterStream &ps) {
  o << "Stream<" << ::getMangledName(item, ps) << ">";
}

Type RustStreamType::getType() const { return getImpl()->getType(); }

Type RustStreamTypeStorage::getType() const { return item; }

//===----------------------------------------------------------------------===//
// RustSinkStreamType
//===----------------------------------------------------------------------===//
struct RustSinkStreamTypeStorage : public TypeStorage {
  RustSinkStreamTypeStorage(Type item) : item(item) {}

  Type item;

  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == KeyTy(item); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustSinkStreamTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    return new (allocator.allocate<RustSinkStreamTypeStorage>())
        RustSinkStreamTypeStorage(key);
  }

  void printAsMLIR(DialectAsmPrinter &os) const;
  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);

  Type getType() const;
  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;
};

std::string RustSinkStreamType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string
RustSinkStreamTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string buffer;
  std::string mn = ::getMangledName(item, ps);

  llvm::raw_string_ostream mangled(buffer);
  mangled << "PushableStream" << mn.size() << mn << "End";
  mangledName = mangled.str();
  return mangledName;
}

RustSinkStreamType RustSinkStreamType::get(RustDialect *dialect, Type item) {
  return Base::get(dialect->getContext(), item);
}

void RustSinkStreamType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustSinkStreamTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "<";
  ::printAsMLIR(item, os);
  os << ">";
}

void RustSinkStreamTypeStorage::printAsRust(llvm::raw_ostream &o,
                                            rust::RustPrinterStream &ps) {
  getMangledName(ps);
  o << "Pushable<" << ::getMangledName(item, ps) << ">";
}

void RustSinkStreamType::printAsRust(llvm::raw_ostream &o,
                                     rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

Type RustSinkStreamType::getType() const { return getImpl()->getType(); }

Type RustSinkStreamTypeStorage::getType() const { return item; }

//===----------------------------------------------------------------------===//
// RustSourceStreamType
//===----------------------------------------------------------------------===//
struct RustSourceStreamTypeStorage : public TypeStorage {
  RustSourceStreamTypeStorage(Type item) : item(item) {}

  Type item;

  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == KeyTy(item); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustSourceStreamTypeStorage *construct(TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
    return new (allocator.allocate<RustSourceStreamTypeStorage>())
        RustSourceStreamTypeStorage(key);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);

  void printAsMLIR(DialectAsmPrinter &os) const;

  Type getType() const;

  std::string getMangledName(rust::RustPrinterStream &ps);
};

std::string RustSourceStreamType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string
RustSourceStreamTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  return "<a stream should not be output as rust>";
}

RustSourceStreamType RustSourceStreamType::get(RustDialect *dialect,
                                               Type item) {
  return Base::get(dialect->getContext(), item);
}

void RustSourceStreamType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustSourceStreamTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "<";
  ::printAsMLIR(item, os);
  os << ">";
}

void RustSourceStreamType::printAsRust(llvm::raw_ostream &o,
                                       rust::RustPrinterStream &ps) {
  return getImpl()->printAsRust(o, ps);
}

void RustSourceStreamTypeStorage::printAsRust(llvm::raw_ostream &o,
                                              rust::RustPrinterStream &ps) {
  o << "Pullable<" << ::getMangledName(item, ps) << ">";
}

Type RustSourceStreamType::getType() const { return getImpl()->getType(); }

Type RustSourceStreamTypeStorage::getType() const { return item; }

//===----------------------------------------------------------------------===//
// RustStructType
//===----------------------------------------------------------------------===//

struct RustStructTypeStorage : public TypeStorage {
  RustStructTypeStorage(bool isCompact,
                        ArrayRef<RustStructType::StructFieldTy> fields)
      : structFields(fields.begin(), fields.end()), compact(isCompact) {}

  SmallVector<RustStructType::StructFieldTy, 4> structFields;

  using KeyTy = std::pair<bool, ArrayRef<RustStructType::StructFieldTy>>;

  bool operator==(const KeyTy &key) const {
    KeyTy self(compact, structFields);
    return key.first == self.first && key.second == self.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(llvm::hash_value(key.first),
                              llvm::hash_value(key.second));
  }

  static RustStructTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<RustStructTypeStorage>())
        RustStructTypeStorage(key.first, key.second);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);

  void printAsMLIR(DialectAsmPrinter &os) const;

  unsigned getNumFields() const;
  StringRef getFieldName(unsigned idx) const;
  Type getFieldType(unsigned idx) const;
  bool isCompact() const { return compact; }

  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  bool compact;
  std::string mangledName;
};

RustStructType RustStructType::get(RustDialect *dialect, bool isCompact,
                                   ArrayRef<StructFieldTy> fields) {
  return Base::get(dialect->getContext(), isCompact, fields);
}

void RustStructTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "struct<";
  if (isCompact())
    os << "<";
  for (unsigned i = 0; i < structFields.size(); i++) {
    if (i != 0)
      os << ", ";
    os << structFields[i].first.getValue();
    os << " : ";
    ::printAsMLIR(structFields[i].second, os);
  }
  if (isCompact())
    os << "<";
  os << ">";
}

void RustStructType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustStructType::printAsRust(llvm::raw_ostream &o,
                                 rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

unsigned RustStructType::getNumFields() const {
  return getImpl()->getNumFields();
}

unsigned RustStructTypeStorage::getNumFields() const {
  return structFields.size();
}

StringRef RustStructType::getFieldName(unsigned idx) const {
  return getImpl()->getFieldName(idx);
}

StringRef RustStructTypeStorage::getFieldName(unsigned idx) const {
  return structFields[idx].first.getValue();
}

Type RustStructType::getFieldType(unsigned idx) const {
  return getImpl()->getFieldType(idx);
}

Type RustStructTypeStorage::getFieldType(unsigned idx) const {
  return structFields[idx].second;
}

bool RustStructType::isCompact() const { return getImpl()->isCompact(); }

std::string RustStructType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string RustStructTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);

  if (isCompact())
    mangled << "Compact";
  mangled << "Struct";

  for (auto &f : structFields) {
    StringRef fieldName = f.first.getValue();
    std::string mn = ::getMangledName(f.second, ps);
    mangled << fieldName.size() << fieldName;
    mangled << mn.size() << mn;
  }
  mangled << "End";
  mangledName = mangled.str();

  llvm::raw_ostream &tyStream = ps.getNamedTypesStream();

  if (isCompact())
    tyStream << "#[rewrite(compact)]\n";
  else
    tyStream << "#[rewrite]\n";
  tyStream << "pub struct " << mangledName << " {\n  ";

  for (unsigned i = 0; i < structFields.size(); i++) {
    if (i != 0)
      tyStream << ",\n  ";

    tyStream << "  " << structFields[i].first.getValue() << " : ";
    tyStream << ::getMangledName(structFields[i].second, ps);
  }
  tyStream << "\n}\n";

  return mangledName;
}

void RustStructTypeStorage::printAsRust(llvm::raw_ostream &o,
                                        rust::RustPrinterStream &ps) {
  o << getMangledName(ps);
}

//===----------------------------------------------------------------------===//
// RustGenericADTType
//===----------------------------------------------------------------------===//

struct RustGenericADTTypeStorage : public TypeStorage {
  using KeyTy = std::pair<std::string, llvm::ArrayRef<mlir::Type>>;

  RustGenericADTTypeStorage(StringRef name, ArrayRef<Type> parms)
      : name(name), parameters(parms.begin(), parms.end()) {}

  std::string name;
  SmallVector<Type, 4> parameters;

  bool operator==(const KeyTy &key) const {
    return KeyTy(name, parameters) == key;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustGenericADTTypeStorage *construct(TypeStorageAllocator &allocator,
                                              KeyTy key) {
    return new (allocator.allocate<RustGenericADTTypeStorage>())
        RustGenericADTTypeStorage(key.first, key.second);
  }

  void printAsRust(raw_ostream &os, rust::RustPrinterStream &ps);
  void printAsMLIR(DialectAsmPrinter &os) const;

  void emitNestedTypedefs(rust::RustPrinterStream &os) const;
  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;
};

RustGenericADTType RustGenericADTType::get(RustDialect *dialect,
                                           StringRef templateName,
                                           ArrayRef<Type> parameters) {
  return Base::get(dialect->getContext(), templateName, parameters);
}

void RustGenericADTType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustGenericADTType::printAsRust(raw_ostream &os,
                                     rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(os, ps);
}

std::string RustGenericADTType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

// Convert a string to a valid identifier
static std::string mangleRustTypeToIdentifier(const std::string &in) {
  std::string n;
  for (size_t i = 0; i != in.size(); i++) {
    switch (in[i]) {
    case '\'':
      n.append("tick");
      break;
    case '&':
      n.append("ampersand");
      break;
    case ' ':
      n.append("blank");
      break;
    case ':':
      n.append("colon");
      break;
    default:
      n.push_back(in[i]);
    }
  }
  return n;
}

std::string
RustGenericADTTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string n = ::mangleRustTypeToIdentifier(name);

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);
  mangled << "ADT" << n.size() << n;

  for (auto &p : parameters) {
    std::string tmp = ::getMangledName(p, ps);
    mangled << tmp.size() << tmp;
  }
  mangled << "End";
  mangledName = mangled.str();

  llvm::raw_ostream &tyStream = ps.getNamedTypesStream();

  tyStream << "type " << mangledName << " = " << name << "<";
  bool first = true;
  for (auto &p : parameters) {
    if (!first)
      tyStream << ", ";
    tyStream << ::getMangledName(p, ps);
    first = false;
  }
  tyStream << ">;\n";

  return mangledName;
}

void RustGenericADTTypeStorage::printAsRust(llvm::raw_ostream &o,
                                            rust::RustPrinterStream &ps) {
  o << getMangledName(ps);
}

void RustGenericADTTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  std::string str;
  llvm::raw_string_ostream s(str);

  os << name << "<";
  bool first = true;
  for (auto &p : parameters) {
    if (!first)
      os << ", ";
    ::printAsMLIR(p, os);
    first = false;
  }
  os << ">";
}

//===----------------------------------------------------------------------===//
// RustTensorType
//===----------------------------------------------------------------------===//

struct RustTensorTypeStorage : public TypeStorage {
  using KeyTy = std::pair<Type, ArrayRef<int64_t>>;

  RustTensorTypeStorage(KeyTy key)
      : elementTy(key.first), dimensions(key.second.begin(), key.second.end()) {
  }

  Type elementTy;
  SmallVector<int64_t, 3> dimensions;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementTy, dimensions);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<RustTensorTypeStorage>())
        RustTensorTypeStorage(key);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);
  void printAsMLIR(DialectAsmPrinter &os) const;

  ArrayRef<int64_t> getDimensions() const { return dimensions; }
  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;
};

RustTensorType RustTensorType::get(RustDialect *dialect, Type elementTy,
                                   ArrayRef<int64_t> dimensions) {
  return Base::get(dialect->getContext(), elementTy, dimensions);
}

void RustTensorTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "tensor<";
  for (unsigned i = 0; i < dimensions.size(); i++)
    os << dimensions[i] << "x";
  ::printAsMLIR(elementTy, os);
  os << ">";
}

void RustTensorType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustTensorType::printAsRust(llvm::raw_ostream &o,
                                 rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

std::string RustTensorType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string RustTensorTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);
  mangled << "Tensor";
  for (unsigned i = 0; i < dimensions.size(); i++)
    mangled << dimensions[i] << "x";
  mangled << ::getMangledName(elementTy, ps);
  mangledName = mangled.str();

  llvm::raw_ostream &tyStream = ps.getNamedTypesStream();

  tyStream << "type " << mangledName << " = "
           << "Rc<Array<" << ::getMangledName(elementTy, ps) << ", Dim<[Ix; "
           << dimensions.size() << "]>>>\n";

  return mangledName;
}

void RustTensorTypeStorage::printAsRust(llvm::raw_ostream &o,
                                        rust::RustPrinterStream &ps) {
  ps.registerDirective("rc-import", "use std::rc::Rc;\n");
  ps.registerDirective("ndarray-import", "use ndarray::{Array,Dim,Ix};\n");
  ps.registerDependency("ndarray",
                        (Twine("\"") + CrateVersions::ndarray + "\"").str());

  o << getMangledName(ps);
}

ArrayRef<int64_t> RustTensorType::getDimensions() const {
  return getImpl()->getDimensions();
}

//===----------------------------------------------------------------------===//
// RustTupleType
//===----------------------------------------------------------------------===//

struct RustTupleTypeStorage : public TypeStorage {
  RustTupleTypeStorage(ArrayRef<Type> fields)
      : tupleFields(fields.begin(), fields.end()) {}

  SmallVector<Type, 4> tupleFields;

  using KeyTy = ArrayRef<Type>;

  bool operator==(const KeyTy &key) const { return key == KeyTy(tupleFields); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustTupleTypeStorage *construct(TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<RustTupleTypeStorage>())
        RustTupleTypeStorage(key);
  }

  void printAsRust(llvm::raw_ostream &o, rust::RustPrinterStream &ps);
  void printAsMLIR(DialectAsmPrinter &os) const;

  std::string getMangledName(rust::RustPrinterStream &ps);

private:
  std::string mangledName;
};

RustTupleType RustTupleType::get(RustDialect *dialect, ArrayRef<Type> fields) {
  return Base::get(dialect->getContext(), fields);
}

void RustTupleTypeStorage::printAsMLIR(DialectAsmPrinter &os) const {
  os << "tuple<";
  for (unsigned i = 0; i < tupleFields.size(); i++) {
    if (i != 0)
      os << ", ";
    ::printAsMLIR(tupleFields[i], os);
  }
  os << ">";
}

void RustTupleType::printAsMLIR(DialectAsmPrinter &os) const {
  getImpl()->printAsMLIR(os);
}

void RustTupleType::printAsRust(llvm::raw_ostream &o,
                                rust::RustPrinterStream &ps) {
  getImpl()->printAsRust(o, ps);
}

std::string RustTupleType::getMangledName(rust::RustPrinterStream &ps) {
  return getImpl()->getMangledName(ps);
}

std::string RustTupleTypeStorage::getMangledName(rust::RustPrinterStream &ps) {
  if (!mangledName.empty())
    return mangledName;

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);
  mangled << "Tuple";

  for (auto &f : tupleFields) {
    std::string tmp = ::getMangledName(f, ps);
    mangled << tmp.size() << tmp;
  }
  mangled << "End";
  mangledName = mangled.str();

  llvm::raw_ostream &tyStream = ps.getNamedTypesStream();

  tyStream << "type " << mangledName << " = "
           << "(";
  for (unsigned i = 0; i < tupleFields.size(); i++) {
    if (i != 0)
      tyStream << ",\n";
    tyStream << ::getMangledName(tupleFields[i], ps);
  }
  tyStream << ");\n";

  return mangledName;
}

void RustTupleTypeStorage::printAsRust(llvm::raw_ostream &o,
                                       rust::RustPrinterStream &ps) {
  ps.registerDirective("rc-import", "use std::rc::Rc;\n");
  ps << getMangledName(ps);
}

} // namespace types

static bool isAnyRustType(Type type) {
  if (type.isa<RustType>() || type.isa<RustStructType>() ||
      type.isa<RustTupleType>() || type.isa<RustTensorType>() ||
      type.isa<RustEnumType>() || type.isa<RustStreamType>() ||
      type.isa<RustSinkStreamType>() || type.isa<RustSourceStreamType>() ||
      type.isa<RustGenericADTType>())
    return true;
  if (type.isa<FunctionType>())
    return isRustFunctionType(type);
  return false;
}

bool isRustFunctionType(Type type) {
  if (FunctionType fty = type.dyn_cast<FunctionType>()) {
    for (Type t : fty.getInputs())
      if (!isAnyRustType(t))
        return false;
    for (Type t : fty.getResults())
      if (!isAnyRustType(t))
        return false;
    return true;
  }
  return false;
}

RustPrinterStream &operator<<(RustPrinterStream &os, const Type &t) {
  llvm::raw_ostream &body = os.getBodyStream();
  ::printAsRust(t, body, os);
  return os;
}

RustPrinterStream &operator<<(RustPrinterStream &os, const llvm::StringRef &s) {
  llvm::raw_ostream &body = os.getBodyStream();
  body << s;
  return os;
}

RustPrinterStream &operator<<(RustPrinterStream &os, uint64_t u) {
  llvm::raw_ostream &body = os.getBodyStream();
  body << u;
  return os;
}

void RustPrinterStream::printAsRust(llvm::raw_ostream &o, FunctionType fTy) {
  o << ::getMangledName(fTy, *this);
}

void RustPrinterStream::printType(llvm::raw_ostream &o, Type t) {
  ::printAsRust(t, o, *this);
}

RustPrinterStream &operator<<(RustPrinterStream &ps, const Value &v) {
  ps.getBodyStream() << ps.get(v);
  return ps;
}

std::string RustPrinterStream::getMangledName(FunctionType fTy) {
  auto found = FunctionTypes.find(fTy);
  if (found != FunctionTypes.end())
    return found->second;

  std::string buffer;
  llvm::raw_string_ostream mangled(buffer);

  mangled << "Function";
  for (Type i : fTy.getInputs()) {
    auto iname = ::getMangledName(i, *this);
    mangled << iname.size() << iname;
  }
  if (fTy.getNumResults())
    mangled << "R" << ::getMangledName(fTy.getResult(0), *this);
  auto mn = mangled.str();

  FunctionTypes[fTy] = mn;
  llvm::raw_ostream &tyStream = getNamedTypesStream();

  tyStream << "type " << mn << " = function!((";
  for (Type t : fTy.getInputs())
    tyStream << ::getMangledName(t, *this) << ", ";
  tyStream << ")";
  if (fTy.getNumResults())
    tyStream << " -> " << ::getMangledName(fTy.getResult(0), *this);
  tyStream << ");\n";

  return mn;
}

} // namespace rust

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Rust/Rust.cpp.inc"
