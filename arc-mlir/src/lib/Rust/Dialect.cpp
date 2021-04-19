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

static bool outputIsToModule() { return !rustModuleFile.getValue().empty(); }

//===----------------------------------------------------------------------===//
// RustDialect
//===----------------------------------------------------------------------===//

void RustDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Rust/Rust.cpp.inc"
      >();
  addTypes<RustType>();
  addTypes<RustStructType>();
  addTypes<RustTensorType>();
  addTypes<RustTupleType>();

  auto ctx = getContext();

  floatTy = RustType::get(ctx, "f32");
  doubleTy = RustType::get(ctx, "f64");
  float16Ty = RustType::get(ctx, "arcorn::f16");
  bFloat16Ty = RustType::get(ctx, "arcorn::bf16");
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
  if (auto t = type.dyn_cast<RustType>())
    t.print(os);
  else if (auto t = type.dyn_cast<RustStructType>())
    t.print(os);
  else if (auto t = type.dyn_cast<RustTensorType>())
    t.print(os);
  else if (auto t = type.dyn_cast<RustTupleType>())
    t.print(os);
  else
    llvm_unreachable("Unhandled Rust type");
}

struct CloneControl {
  const Value V;

public:
  CloneControl(const Value v) : V(v) {}
  virtual ~CloneControl() {}
  virtual void output(llvm::raw_string_ostream &os) const = 0;

  bool needsClone() const {
    const Type t = V.getType();
    if (t.isa<rust::types::RustStructType>())
      return true;
    if (t.isa<rust::types::RustTensorType>())
      return true;
    if (t.isa<rust::types::RustTupleType>())
      return true;
    return false;
  }
};

struct CloneStart : public CloneControl {
  CloneStart(Value v) : CloneControl(v) {}
  void output(llvm::raw_string_ostream &os) const override {
    os << "Rc::clone(&";
  }
};

struct CloneEnd : public CloneControl {
  CloneEnd(Value v) : CloneControl(v) {}
  void output(llvm::raw_string_ostream &os) const override { os << ")"; };
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

/// Hook for FunctionLike verifier.
LogicalResult RustExtFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  Attribute lang = (*this)->getAttr("language");
  if (!lang || !lang.isa<StringAttr>())
    return emitOpError("requires 'language' attribute");
  StringAttr lang_str = lang.cast<StringAttr>();
  if (!lang_str.getValue().equals("rust"))
    return emitOpError("only supports the Rust language");
  Attribute crate = (*this)->getAttr("crate");
  if (!crate || !crate.isa<StringAttr>())
    return emitOpError("requires 'crate' attribute");

  std::string dependency_key =
      ("rust.dependency." + crate.cast<StringAttr>().getValue()).str();

  Attribute dep = (*this)->getParentOp()->getAttr(dependency_key);
  if (!dep || !dep.isa<StringAttr>())
    return (*this)->getParentOp()->emitOpError("requires '" + dependency_key +
                                               "' attribute");

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
  RustFuncOp function = returnOp->getParentOfType<RustFuncOp>();
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

RustPrinterStream &operator<<(RustPrinterStream &os, const Type &type) {
  if (auto t = type.dyn_cast<RustType>())
    os.print(t.getRustType());
  else if (auto t = type.dyn_cast<RustStructType>())
    os.print(t);
  else if (auto t = type.dyn_cast<RustTensorType>())
    t.printAsRust(os);
  else if (auto t = type.dyn_cast<RustTupleType>())
    t.printAsRust(os);
  else
    os << "<not-a-rust-type>";
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
  if (PS.hasTypesOutput())
    out << crateName << "_types = { path = \"../" << crateName
        << "_types\", version = \"0.1.0\" }\n";
  out.close();
  return true;
}

static bool writeTypesToml(StringRef filename, StringRef crateName,
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
      << "name = \"" << crateName << "_types\"\n"
      << "version = \"0.1.0\"\n"
      << "authors = [\"arc-mlir\"]\n"
      << "edition = \"2018\"\n"
      << "\n"
      << "[dependencies]\n";

  out.close();
  return true;
}

LogicalResult rust::writeModuleAsCrates(ModuleOp module, std::string top_dir,
                                        std::string rustTrailer,
                                        llvm::raw_ostream &o) {

  // Create the files for the main rust crate
  StringRef crateName = module.getName().getValueOr("unknown");

  if (!crateNameOverride.getValue().empty())
    crateName = crateNameOverride;

  llvm::errs() << "writing crate \"" << crateName << "\" to " << top_dir
               << "\n";

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

  // Create the files for the type crate
  std::string typesCrateName = crateName.str() + "_types";
  SmallString<128> types_crate_dir(top_dir);
  llvm::sys::path::append(types_crate_dir, typesCrateName);
  SmallString<128> types_src_dir(types_crate_dir);
  llvm::sys::path::append(types_src_dir, "src");

  SmallString<128> types_toml_filename(types_crate_dir);
  llvm::sys::path::append(types_toml_filename, "Cargo.toml");

  SmallString<128> types_rs_filename(types_src_dir);
  llvm::sys::path::append(types_rs_filename, "lib.rs");

  EC = llvm::sys::fs::create_directories(types_src_dir);
  if (EC) {
    llvm::errs() << "Unable to create crate directories: " << types_src_dir
                 << ", " << EC.message() << ".\n";
    return failure();
  }

  llvm::raw_fd_ostream types(types_rs_filename, EC,
                             llvm::sys::fs::CD_CreateAlways,
                             llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Failed to create " << types_rs_filename << ", "
                 << EC.message() << "\n";
    return failure();
  }

  RustPrinterStream PS(out, types, crateName.str());

  for (Operation &operation : module) {
    if (RustFuncOp op = dyn_cast<RustFuncOp>(operation))
      op.writeRust(PS);
    else if (RustExtFuncOp op = dyn_cast<RustExtFuncOp>(operation))
      op.writeRust(PS);
  }

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
  types.close();

  if (!writeToml(toml_filename, crateName, PS))
    return failure();

  if (!writeTypesToml(types_toml_filename, crateName, PS))
    return failure();

  return success();
}

LogicalResult rust::writeModuleAsInline(ModuleOp module, llvm::raw_ostream &o) {
  std::string ms, ts;
  llvm::raw_string_ostream m(ms), t(ts);

  RustPrinterStream PS(m, t, "cratename", true);

  for (Operation &operation : module) {
    if (RustFuncOp op = dyn_cast<RustFuncOp>(operation))
      op.writeRust(PS);
    else if (RustExtFuncOp op = dyn_cast<RustExtFuncOp>(operation))
      op.writeRust(PS);
  }

  PS.flush();
  m.flush();
  t.flush();

  o.write(ts.data(), ts.size());
  o.write(ms.data(), ms.size());

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

// Write this function as Rust code to os
void RustExtFuncOp::writeRust(RustPrinterStream &PS) {
  StringRef crate = (*this)->getAttrOfType<StringAttr>("crate").getValue();
  StringRef symbol = (*this)->getAttrOfType<StringAttr>("sym_name").getValue();

  std::string d =
      ("use " + crate + "::" + symbol + " as " + symbol + ";\n").str();
  PS.registerDirective(d, d);

  std::string dependency_key = ("rust.dependency." + crate).str();
  PS.registerDependency(crate.str(),
                        (*this)
                            ->getParentOp()
                            ->getAttrOfType<StringAttr>(dependency_key)
                            .getValue()
                            .str());
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

void RustBinaryRcOp::writeRust(RustPrinterStream &PS) {
  auto r = getResult();
  PS << "let " << r << ":" << r.getType() << " = Rc::new(&*" << LHS() << " "
     << getOperator() << " &*" << RHS() << ");\n";
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
const char *CrateVersions::hexf = "0.2.1";
const char *CrateVersions::ndarray = "0.13.0";
} // namespace rust

//===----------------------------------------------------------------------===//
// Rust types
//===----------------------------------------------------------------------===//
namespace rust {
namespace types {

static std::string getTypeString(Type type) {
  if (auto t = type.dyn_cast<RustStructType>())
    return t.getRustType();
  if (auto t = type.dyn_cast<RustTupleType>())
    return t.getRustType();
  if (auto t = type.dyn_cast<RustTensorType>())
    return t.getRustType();
  if (auto t = type.dyn_cast<RustType>())
    return t.getRustType().str();
  return "<unsupported type>";
}

static std::string getTypeSignature(Type type) {
  if (auto t = type.dyn_cast<RustStructType>())
    return t.getSignature();
  if (auto t = type.dyn_cast<RustTupleType>())
    return t.getSignature();
  if (auto t = type.dyn_cast<RustTensorType>())
    return t.getSignature();
  if (auto t = type.dyn_cast<RustType>())
    return t.getSignature();
  return "<unsupported type>";
}

struct RustTypeStorage : public TypeStorage {
  RustTypeStorage(std::string type) : rustType(type) {}

  std::string rustType;

  using KeyTy = std::string;

  bool operator==(const KeyTy &key) const { return key == KeyTy(rustType); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<RustTypeStorage>()) RustTypeStorage(key);
  }

  raw_ostream &printAsRust(raw_ostream &os) const {
    os << rustType;
    return os;
  }
  std::string getSignature() const;
};

RustType RustType::get(MLIRContext *context, StringRef type) {
  return Base::get(context, type);
}

StringRef RustType::getRustType() const { return getImpl()->rustType; }

void RustType::print(DialectAsmPrinter &os) const { os << getRustType(); }

raw_ostream &RustType::printAsRust(raw_ostream &os) const {
  return getImpl()->printAsRust(os);
}

bool RustType::isBool() const { return getRustType().equals("bool"); }

RustType RustType::getFloatTy(RustDialect *dialect) { return dialect->floatTy; }

RustType RustType::getDoubleTy(RustDialect *dialect) {
  return dialect->doubleTy;
}

RustType RustType::getFloat16Ty(RustDialect *dialect) {
  return dialect->float16Ty;
}

RustType RustType::getBFloat16Ty(RustDialect *dialect) {
  return dialect->bFloat16Ty;
}

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

std::string RustType::getSignature() const { return getImpl()->getSignature(); }

std::string RustTypeStorage::getSignature() const { return rustType; }

//===----------------------------------------------------------------------===//
// RustStructType
//===----------------------------------------------------------------------===//

struct RustStructTypeStorage : public TypeStorage {
  RustStructTypeStorage(ArrayRef<RustStructType::StructFieldTy> fields,
                        unsigned id)
      : structFields(fields.begin(), fields.end()), id(id) {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "ArcStruct";

    for (auto &f : fields) {
      s << "F" << f.first.getValue() << "T";
      s << getTypeSignature(f.second);
    }
    signature = s.str();
  }

  SmallVector<RustStructType::StructFieldTy, 4> structFields;
  unsigned id;

  using KeyTy = ArrayRef<RustStructType::StructFieldTy>;

  bool operator==(const KeyTy &key) const { return key == KeyTy(structFields); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static RustStructTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<RustStructTypeStorage>())
        RustStructTypeStorage(key, idCounter++);
  }

  RustPrinterStream &printAsRust(RustPrinterStream &os) const;
  raw_ostream &printAsRustNamedType(raw_ostream &os) const;
  void print(DialectAsmPrinter &os) const { os << getRustType(); }
  StringRef getFieldName(unsigned idx) const;

  std::string getRustType() const;
  unsigned getStructTypeId() const;

  void emitNestedTypedefs(rust::RustPrinterStream &os) const;
  std::string getSignature() const;

private:
  static unsigned idCounter;
  std::string signature;
};

unsigned RustStructTypeStorage::idCounter = 0;

RustStructType RustStructType::get(RustDialect *dialect,
                                   ArrayRef<StructFieldTy> fields) {
  mlir::MLIRContext *ctx = fields.front().second.getContext();
  return Base::get(ctx, fields);
}

void RustStructType::print(DialectAsmPrinter &os) const {
  getImpl()->print(os);
}

RustPrinterStream &RustStructType::printAsRust(RustPrinterStream &os) const {
  return getImpl()->printAsRust(os);
}

raw_ostream &RustStructType::printAsRustNamedType(raw_ostream &os) const {
  return getImpl()->printAsRustNamedType(os);
}

StringRef RustStructType::getFieldName(unsigned idx) const {
  return getImpl()->getFieldName(idx);
}

StringRef RustStructTypeStorage::getFieldName(unsigned idx) const {
  return structFields[idx].first.getValue();
}

std::string RustStructType::getRustType() const {
  return getImpl()->getRustType();
}

unsigned RustStructTypeStorage::getStructTypeId() const { return id; }

unsigned RustStructType::getStructTypeId() const {
  return getImpl()->getStructTypeId();
}
std::string RustStructTypeStorage::getRustType() const { return signature; }

RustPrinterStream &
RustStructTypeStorage::printAsRust(RustPrinterStream &ps) const {

  llvm::raw_ostream &os = ps.getNamedTypesStream();
  llvm::raw_ostream &uses_os = ps.getUsesStream();
  // First ensure that any structs used by this struct are defined
  emitNestedTypedefs(ps);

  os << "pub struct ";
  printAsRustNamedType(os) << "Value {\n  ";

  for (unsigned i = 0; i < structFields.size(); i++) {
    if (i != 0)
      os << ",\n  ";
    os << "pub " << structFields[i].first.getValue() << " : ";
    os << getTypeString(structFields[i].second);
  }
  os << "\n}\n";
  os << "pub type ";
  printAsRustNamedType(os) << " = Rc<";
  printAsRustNamedType(os) << "Value>;\n";
  ps.registerDirective("rc-import", "use std::rc::Rc;\n");
  uses_os << "use ";
  ps.printModuleName(uses_os) << "_types::";
  printAsRustNamedType(uses_os) << " as ";
  printAsRustNamedType(uses_os) << ";\n";
  uses_os << "use ";
  ps.printModuleName(uses_os) << "_types::";
  printAsRustNamedType(uses_os) << "Value as ";
  printAsRustNamedType(uses_os) << "Value;\n";
  return ps;
}

void RustStructType::emitNestedTypedefs(rust::RustPrinterStream &ps) const {
  return getImpl()->emitNestedTypedefs(ps);
}

void RustStructTypeStorage::emitNestedTypedefs(
    rust::RustPrinterStream &ps) const {
  // First ensure that any structs used by this tuple are defined
  for (unsigned i = 0; i < structFields.size(); i++)
    if (structFields[i].second.isa<RustStructType>())
      ps.writeStructDefiniton(structFields[i].second.cast<RustStructType>());
    else if (structFields[i].second.isa<RustTupleType>())
      structFields[i].second.cast<RustTupleType>().emitNestedTypedefs(ps);
}

raw_ostream &
RustStructTypeStorage::printAsRustNamedType(raw_ostream &os) const {

  os << signature;
  return os;
}

std::string RustStructType::getSignature() const {
  return getImpl()->getSignature();
}

std::string RustStructTypeStorage::getSignature() const { return signature; }

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

  RustPrinterStream &printAsRust(RustPrinterStream &os) const;
  void print(DialectAsmPrinter &os) const { os << getRustType(); }

  std::string getRustType() const;

  ArrayRef<int64_t> getDimensions() const { return dimensions; }
  std::string getSignature() const;
};

RustTensorType RustTensorType::get(RustDialect *dialect, Type elementTy,
                                   ArrayRef<int64_t> dimensions) {
  mlir::MLIRContext *ctx = elementTy.getContext();
  return Base::get(ctx, elementTy, dimensions);
}

void RustTensorType::print(DialectAsmPrinter &os) const {
  getImpl()->print(os);
}

RustPrinterStream &RustTensorType::printAsRust(RustPrinterStream &os) const {
  return getImpl()->printAsRust(os);
}

std::string RustTensorType::getRustType() const {
  return getImpl()->getRustType();
}

std::string RustTensorTypeStorage::getRustType() const {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "Rc<Array<" << getTypeString(elementTy) << ", Dim<[Ix; "
    << dimensions.size() << "]>>>";
  return s.str();
}

RustPrinterStream &
RustTensorTypeStorage::printAsRust(RustPrinterStream &ps) const {
  ps.registerDirective("rc-import", "use std::rc::Rc;\n");
  ps.registerDirective("ndarray-import", "use ndarray::{Array,Dim,Ix};\n");
  ps.registerDependency("ndarray",
                        (Twine("\"") + CrateVersions::ndarray + "\"").str());
  ps << getRustType();
  return ps;
}

ArrayRef<int64_t> RustTensorType::getDimensions() const {
  return getImpl()->getDimensions();
}

std::string RustTensorType::getSignature() const {
  return getImpl()->getSignature();
}

std::string RustTensorTypeStorage::getSignature() const {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "TensorT" << getTypeSignature(elementTy) << "x" << dimensions.size();
  return s.str();
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

  RustPrinterStream &printAsRust(RustPrinterStream &os) const;
  void print(DialectAsmPrinter &os) const { os << getRustType(); }

  std::string getRustType() const;

  void emitNestedTypedefs(rust::RustPrinterStream &ps) const;

  std::string getSignature() const;
};

RustTupleType RustTupleType::get(RustDialect *dialect, ArrayRef<Type> fields) {
  mlir::MLIRContext *ctx = fields.front().getContext();
  return Base::get(ctx, fields);
}

void RustTupleType::print(DialectAsmPrinter &os) const { getImpl()->print(os); }

RustPrinterStream &RustTupleType::printAsRust(RustPrinterStream &os) const {
  return getImpl()->printAsRust(os);
}

std::string RustTupleType::getRustType() const {
  return getImpl()->getRustType();
}

std::string RustTupleTypeStorage::getRustType() const {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "Rc<(";
  for (unsigned i = 0; i < tupleFields.size(); i++)
    s << getTypeString(tupleFields[i]) << ", ";
  s << ")>";
  return s.str();
}

RustPrinterStream &
RustTupleTypeStorage::printAsRust(RustPrinterStream &ps) const {
  emitNestedTypedefs(ps);

  ps.registerDirective("rc-import", "use std::rc::Rc;\n");
  ps << getRustType();
  return ps;
}

void RustTupleType::emitNestedTypedefs(rust::RustPrinterStream &ps) const {
  return getImpl()->emitNestedTypedefs(ps);
}

void RustTupleTypeStorage::emitNestedTypedefs(
    rust::RustPrinterStream &ps) const {
  // First ensure that any structs used by this tuple are defined
  for (unsigned i = 0; i < tupleFields.size(); i++)
    if (tupleFields[i].isa<RustStructType>())
      ps.writeStructDefiniton(tupleFields[i].cast<RustStructType>());
    else if (tupleFields[i].isa<RustTupleType>())
      tupleFields[i].cast<RustTupleType>().emitNestedTypedefs(ps);
}

std::string RustTupleType::getSignature() const {
  return getImpl()->getSignature();
}

std::string RustTupleTypeStorage::getSignature() const {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "Tuple";
  for (unsigned i = 0; i < tupleFields.size(); i++)
    s << "T" << getTypeSignature(tupleFields[i]);
  return s.str();
}

} // namespace types
} // namespace rust

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Rust/Rust.cpp.inc"
