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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace rust;
using namespace types;

//===----------------------------------------------------------------------===//
// RustDialect
//===----------------------------------------------------------------------===//

RustDialect::RustDialect(mlir::MLIRContext *ctx) : mlir::Dialect("rust", ctx) {
  addOperations<
#define GET_OP_LIST
#include "Rust/Rust.cpp.inc"
      >();
  addTypes<RustType>();
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

void RustDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Rust type");
  case RUST_TYPE:
    type.cast<RustType>().print(os);
    break;
  }
}

//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//

LogicalResult CrateOp::customVerify() { return mlir::success(); }

/// Hook for FunctionLike verifier.
LogicalResult RustFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
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

  ArrayRef<Type> funcArgTypes = getType().getInputs();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    Type blockArgType = front().getArgument(i).getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", found "
                           << blockArgType;
  }

  return success();
}

static LogicalResult verify(RustReturnOp returnOp) {
  RustFuncOp function = returnOp.getParentOfType<RustFuncOp>();

  FunctionType funType = function.getType();
  Type funReturnType = funType.getResult(0);
  Type returnType = returnOp.getResult().getType();

  if (funReturnType != returnType) {
    return returnOp.emitOpError(
               "result type does not match the type of the function: expected ")
           << funReturnType << " but found " << returnType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RustDialect Rust Printing
//===----------------------------------------------------------------------===//
namespace rust {
RustPrinterStream &operator<<(RustPrinterStream &os, const Value &v) {
  return os.print(v);
}

RustPrinterStream &operator<<(RustPrinterStream &os, const RustType &t) {
  return os.print(t);
}
} // namespace rust

static bool writeToml(StringRef filename, StringRef crateName) {
  std::error_code EC;
  llvm::raw_fd_ostream out(filename, EC, llvm::sys::fs::CD_CreateAlways,
                           llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Failed to create " << filename << ", " << EC.message()
                 << "\n";
    return false;
  }

  out << "[package]\n"
      << "name = \"" << crateName << "\"\n"
      << "version = \"0.1.0\"\n"
      << "authors = [\"arc-mlir\"]\n"
      << "edition = \"2018\"\n"
      << "\n"
      << "[dependencies]\n";

  out.close();
  return true;
}

LogicalResult CrateOp::writeCrate(std::string top_dir, llvm::raw_ostream &o) {
  llvm::errs() << "writing crate \"" << getName() << "\" to " << top_dir
               << "\n";
  SmallString<128> crate_dir(top_dir);
  llvm::sys::path::append(crate_dir, getName());
  SmallString<128> src_dir(crate_dir);
  llvm::sys::path::append(src_dir, "src");

  SmallString<128> toml_filename(crate_dir);
  llvm::sys::path::append(toml_filename, "Cargo.toml");

  SmallString<128> rs_filename(src_dir);
  llvm::sys::path::append(rs_filename, "lib.rs");

  std::error_code EC = llvm::sys::fs::create_directories(src_dir);
  if (EC) {
    llvm::errs() << "Unable to create crate directories: " << src_dir << ", "
                 << EC.message() << ".\n";
    return failure();
  }

  if (!writeToml(toml_filename, getName()))
    return failure();

  llvm::raw_fd_ostream out(rs_filename, EC, llvm::sys::fs::CD_CreateAlways,
                           llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Failed to create " << rs_filename << ", " << EC.message()
                 << "\n";
    return failure();
  }

  RustPrinterStream PS(out);

  for (Operation &operation : this->body().front())
    if (RustFuncOp op = dyn_cast<RustFuncOp>(operation))
      op.writeRust(PS);

  PS.flush();

  // Write a small dummy test so that even an empty crate compiles and
  // tests successfully.
  out << "#[cfg(test)]\n"
      << "mod tests {\n"
      << "#[test]\n"
      << "fn it_works() {\n"
      << "assert_eq!(true, true);\n"
      << "}\n"
      << "}\n";

  out.close();

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
  else if (RustIfOp op = dyn_cast<RustIfOp>(operation))
    op.writeRust(PS);
  else if (RustBlockResultOp op = dyn_cast<RustBlockResultOp>(operation))
    op.writeRust(PS);
  else {
    PS.getBodyStream() << "\ncompile_error!(\"Unsupported Op: ";
    operation.print(PS.getBodyStream());
    PS.getBodyStream() << "\");\n";
  }
  return PS;
}

// Write this function as Rust code to os
void RustFuncOp::writeRust(RustPrinterStream &PS) {

  PS << "pub fn " << getName() << "(";

  // Dump the function arguments
  unsigned numFuncArguments = getNumArguments();
  for (unsigned i = 0; i < numFuncArguments; i++) {
    if (i != 0)
      PS << ", ";
    Value v = front().getArgument(i);
    RustType t = v.getType().cast<RustType>();
    PS << v << ": " << t;
  }
  PS << ") ";
  if (getNumFuncResults()) { // The return type
    RustType funReturnType = getType().getResult(0).cast<RustType>();
    PS << "-> " << funReturnType << " ";
  }

  // Dumping the body
  PS << "{\n";
  for (Operation &operation : this->body().front()) {
    ::writeRust(operation, PS);
  }
  PS << "}\n";
}

void RustReturnOp::writeRust(RustPrinterStream &PS) {
  PS << "return " << getOperand() << ";\n";
}

void RustConstantOp::writeRust(RustPrinterStream &PS) { PS.getConstant(*this); }

void RustUnaryOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  types::RustType rt = r.getType().cast<types::RustType>();
  PS << "let " << r << ":" << rt << " = " << getOperator() << getOperand()
     << ";\n";
}

void RustBinaryOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  types::RustType rt = r.getType().cast<types::RustType>();
  PS << "let " << r << ":" << rt << " = " << LHS() << " " << getOperator()
     << " " << RHS() << ";\n";
}

void RustIfOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  types::RustType rt = r.getType().cast<types::RustType>();
  PS << "let " << r << ":" << rt << " = if " << getOperand() << " {\n";
  for (Operation &operation : thenRegion().front())
    ::writeRust(operation, PS);
  PS << "} else {\n";
  for (Operation &operation : elseRegion().front())
    ::writeRust(operation, PS);
  PS << "};\n";
}

void RustBlockResultOp::writeRust(RustPrinterStream &PS) {
  PS << getOperand() << "\n";
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Rust/Rust.cpp.inc"
