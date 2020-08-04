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
#include <mlir/IR/Module.h>
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
  addTypes<RustStructType>();
  addTypes<RustTensorType>();
  addTypes<RustTupleType>();

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
  case RUST_STRUCT:
    type.cast<RustStructType>().print(os);
    break;
  case RUST_TENSOR:
    type.cast<RustTensorType>().print(os);
    break;
  case RUST_TUPLE:
    type.cast<RustTupleType>().print(os);
    break;
  }
}

struct CloneControl {
  const Value V;

public:
  CloneControl(const Value v) : V(v) {}
  virtual ~CloneControl() {}
  virtual void output(llvm::raw_string_ostream &os) const = 0;

  bool needsClone() const {
    const Type t = V.getType();
    switch (t.getKind()) {
    case types::RUST_STRUCT:
    case types::RUST_TENSOR:
    case types::RUST_TUPLE:
      return true;
    case types::RUST_TYPE:
      return false;
    default:
      return false;
    }
  }
};

struct CloneStart : public CloneControl {
  CloneStart(Value v) : CloneControl(v) {}
  void output(llvm::raw_string_ostream &os) const { os << "Rc::clone(&"; }
};

struct CloneEnd : public CloneControl {
  CloneEnd(Value v) : CloneControl(v) {}
  void output(llvm::raw_string_ostream &os) const { os << ")"; };
};

llvm::raw_string_ostream &operator<<(llvm::raw_string_ostream &os,
                                     const CloneControl &cc) {
  if (cc.needsClone())
    cc.output(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//

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

  if (funType.getNumResults() == 0 && returnOp.operands())
    return returnOp.emitOpError("cannot return a value from a void function");

  if (!returnOp.operands() && funType.getNumResults())
    return returnOp.emitOpError("operation must return a ")
           << funType.getResult(0) << " value";

  if (!funType.getNumResults())
    return success();

  Type returnType = returnOp.getOperand(0).getType();
  Type funReturnType = funType.getResult(0);

  if (funReturnType != returnType) {
    return returnOp.emitOpError("result type does not match the type of the "
                                "function: expected ")
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

RustPrinterStream &operator<<(RustPrinterStream &os, const Type &t) {
  switch (t.getKind()) {
  case RUST_TYPE:
    os.print(t.cast<RustType>().getRustType());
    break;
  case RUST_STRUCT:
    os.print(t.cast<RustStructType>());
    break;
  case RUST_TENSOR:
    t.cast<RustTensorType>().printAsRust(os);
    break;
  case RUST_TUPLE:
    t.cast<RustTupleType>().printAsRust(os);
    break;
  default:
    os << "<not-a-rust-type>";
  }
  return os;
}
} // namespace rust

static bool writeToml(StringRef filename, StringRef crateName,
                      RustPrinterStream &PS) {
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
  PS.writeTomlDependencies(out);
  out.close();
  return true;
}

LogicalResult rust::writeModuleAsCrate(ModuleOp module, std::string top_dir,
                                       std::string rustTrailer,
                                       llvm::raw_ostream &o) {
  llvm::errs() << "writing crate \"" << module.getName() << "\" to " << top_dir
               << "\n";
  StringRef crateName = module.getName().getValueOr("unknown");
  SmallString<128> crate_dir(top_dir);
  llvm::sys::path::append(crate_dir, crateName);
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

  llvm::raw_fd_ostream out(rs_filename, EC, llvm::sys::fs::CD_CreateAlways,
                           llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Failed to create " << rs_filename << ", " << EC.message()
                 << "\n";
    return failure();
  }

  RustPrinterStream PS(out);

  for (Operation &operation : module)
    if (RustFuncOp op = dyn_cast<RustFuncOp>(operation))
      op.writeRust(PS);

  PS.flush();

  if (!rustTrailer.empty()) {
    using namespace llvm::sys::fs;
    int fd;
    EC = openFileForRead(rustTrailer, fd, OF_None);
    if (EC) {
      llvm::errs() << "Failed to open " << rustTrailer << ", " << EC.message()
                   << "\n";
      return failure();
    }

    constexpr size_t bufSize = 4096;
    std::vector<char> buffer(bufSize);
    int bytesRead = 0;
    for (;;) {
      bytesRead = read(fd, buffer.data(), bufSize);
      if (bytesRead <= 0)
        break;
      out.write(buffer.data(), bytesRead);
    }

    if (bytesRead < 0) {
      llvm::errs() << "Failed to read contents of " << rustTrailer << "\n";
      return failure();
    }
    close(fd);
  }
  out.close();

  if (!writeToml(toml_filename, crateName, PS))
    return failure();

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
  else if (RustCallOp op = dyn_cast<RustCallOp>(operation))
    op.writeRust(PS);
  else if (RustCompOp op = dyn_cast<RustCompOp>(operation))
    op.writeRust(PS);
  else if (RustFieldAccessOp op = dyn_cast<RustFieldAccessOp>(operation))
    op.writeRust(PS);
  else if (RustIfOp op = dyn_cast<RustIfOp>(operation))
    op.writeRust(PS);
  else if (RustBlockResultOp op = dyn_cast<RustBlockResultOp>(operation))
    op.writeRust(PS);
  else if (RustMakeStructOp op = dyn_cast<RustMakeStructOp>(operation))
    op.writeRust(PS);
  else if (RustMethodCallOp op = dyn_cast<RustMethodCallOp>(operation))
    op.writeRust(PS);
  else if (RustTensorOp op = dyn_cast<RustTensorOp>(operation))
    op.writeRust(PS);
  else if (RustTupleOp op = dyn_cast<RustTupleOp>(operation))
    op.writeRust(PS);
  else if (RustDependencyOp op = dyn_cast<RustDependencyOp>(operation))
    PS.registerDependency(op);
  else if (RustModuleDirectiveOp op =
               dyn_cast<RustModuleDirectiveOp>(operation))
    PS.registerDirective(op);
  else {
    operation.emitError("Unsupported operation");
  }
  return PS;
}

void RustCallOp::writeRust(RustPrinterStream &PS) {
  bool has_result = getNumResults();
  if (has_result) {
    auto r = getResult(0);
    PS << "let " << r << ":" << r.getType() << " = " << CloneStart(r);
  }
  PS << getCallee() << "(";
  for (auto a : getOperands())
    PS << CloneStart(a) << a << CloneEnd(a) << ", ";
  PS << ")";
  if (has_result)
    PS << CloneEnd(getResult(0));
  PS << ";\n";
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
    PS << v << ": " << v.getType();
  }
  PS << ") ";
  if (getNumFuncResults()) { // The return type
    PS << "-> " << getType().getResult(0) << " ";
  }

  // Dumping the body
  PS << "{\n";
  for (Operation &operation : this->body().front()) {
    ::writeRust(operation, PS);
  }
  PS << "}\n";
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
  PS << "let " << r << ":" << r.getType() << " = " << CloneStart(r)
     << getOperator() << "(" << CloneStart(getOperand()) << getOperand()
     << CloneEnd(getOperand()) << ")" << CloneEnd(r) << ";\n";
}

void RustMakeStructOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  RustStructType st = r.getType().cast<RustStructType>();
  PS << "let " << r << ":" << st << " = Rc::new(" << st << "Value { ";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    if (i != 0)
      PS << ", ";
    auto v = args[i];
    PS << st.getFieldName(i) << " : " << CloneStart(v) << v << CloneEnd(v);
  }
  PS << "});\n";
}

void RustMethodCallOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType() << " = " << obj() << "."
     << getMethod() << "(";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    if (i != 0)
      PS << ", ";
    auto v = args[i];
    PS << CloneStart(v) << v << CloneEnd(v);
  }
  PS << ");\n";
}

void RustBinaryOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType() << " = " << CloneStart(r) << LHS()
     << " " << getOperator() << " " << RHS() << CloneEnd(r) << ";\n";
}

void RustCompOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType() << " = " << CloneStart(r) << LHS()
     << " " << getOperator() << " " << RHS() << CloneEnd(r) << ";\n";
}

void RustFieldAccessOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType() << " = " << CloneStart(r)
     << aggregate() << "." << getField() << CloneEnd(r) << ";\n";
}

void RustIfOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  // No clone is needed here as it will be inserted by the block
  // result.
  PS << "let " << r << ":" << r.getType() << " = if " << getOperand() << " {\n";
  for (Operation &operation : thenRegion().front())
    ::writeRust(operation, PS);
  PS << "} else {\n";
  for (Operation &operation : elseRegion().front())
    ::writeRust(operation, PS);
  PS << "};\n";
}

void RustBlockResultOp::writeRust(RustPrinterStream &PS) {
  auto r = getOperand();
  PS << CloneStart(r) << r << CloneEnd(r) << "\n";
}

void RustTensorOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType()
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
  PS << "let " << r << ":" << r.getType() << " = Rc::new((";
  auto args = operands();
  for (unsigned i = 0; i < args.size(); i++) {
    auto v = args[i];
    PS << CloneStart(v) << v << CloneEnd(v) << ", ";
  }
  PS << "));\n";
}

//===----------------------------------------------------------------------===//
// Crate versions
//===----------------------------------------------------------------------===//
namespace rust {
const char *CrateVersions::hexf = "0.1.0";
const char *CrateVersions::ndarray = "0.13.0";
} // namespace rust

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Rust/Rust.cpp.inc"
