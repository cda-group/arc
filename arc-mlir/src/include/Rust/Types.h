//===- Dialect definition for the Rust IR --------------------------------===//
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
// Defines the types of the Rust dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RUST_TYPES_H_
#define RUST_TYPES_H_

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

using namespace mlir;

namespace rust {
class RustPrinterStream;
class RustDialect;

namespace types {

//===----------------------------------------------------------------------===//
// Rust Type Storages
//===----------------------------------------------------------------------===//

struct RustTypeStorage;
struct RustStructTypeStorage;
struct RustTupleTypeStorage;
struct RustTensorTypeStorage;

//===----------------------------------------------------------------------===//
// Rust Types
//===----------------------------------------------------------------------===//

class RustType : public Type::TypeBase<RustType, Type, RustTypeStorage> {
public:
  using Base::Base;

  static RustType get(MLIRContext *context, StringRef type);
  void print(DialectAsmPrinter &os) const;
  raw_ostream &printAsRust(raw_ostream &os) const;
  StringRef getRustType() const;
  bool isBool() const;

  static RustType getFloatTy(RustDialect *dialect);
  static RustType getDoubleTy(RustDialect *dialect);
  static RustType getIntegerTy(RustDialect *dialect, IntegerType ty);

  typedef std::pair<mlir::StringAttr, Type> StructFieldTy;

  std::string getSignature() const;
};

class RustStructType
    : public Type::TypeBase<RustStructType, Type, RustStructTypeStorage> {
public:
  using Base::Base;

  void print(DialectAsmPrinter &os) const;
  rust::RustPrinterStream &printAsRust(rust::RustPrinterStream &os) const;
  raw_ostream &printAsRustNamedType(raw_ostream &os) const;
  std::string getRustType() const;
  unsigned getStructTypeId() const;
  StringRef getFieldName(unsigned idx) const;

  typedef std::pair<mlir::StringAttr, Type> StructFieldTy;
  static RustStructType get(RustDialect *dialect,
                            ArrayRef<StructFieldTy> fields);
  void emitNestedTypedefs(rust::RustPrinterStream &ps) const;
  std::string getSignature() const;
};

class RustTensorType
    : public Type::TypeBase<RustTensorType, Type, RustTensorTypeStorage> {
public:
  using Base::Base;

  void print(DialectAsmPrinter &os) const;
  rust::RustPrinterStream &printAsRust(rust::RustPrinterStream &os) const;
  std::string getRustType() const;

  static RustTensorType get(RustDialect *dialect, Type elementTy,
                            ArrayRef<int64_t> dimensions);

  ArrayRef<int64_t> getDimensions() const;
  std::string getSignature() const;
};

class RustTupleType
    : public Type::TypeBase<RustTupleType, Type, RustTupleTypeStorage> {
public:
  using Base::Base;

  void print(DialectAsmPrinter &os) const;
  rust::RustPrinterStream &printAsRust(rust::RustPrinterStream &os) const;
  std::string getRustType() const;

  static RustTupleType get(RustDialect *dialect, ArrayRef<Type> fields);
  void emitNestedTypedefs(rust::RustPrinterStream &ps) const;
  std::string getSignature() const;
};

} // namespace types
} // namespace rust

#endif // RUST_TYPES_H_
