//===- Rust IR Dialect registration in MLIR ------------------------------===//
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

#include "Rust/Types.h"
#include "Rust/Rust.h"
#include "Rust/RustPrinterStream.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace rust;
using namespace types;

namespace rust {
namespace types {

//===----------------------------------------------------------------------===//
// RustType
//===----------------------------------------------------------------------===//

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
};

RustType RustType::get(MLIRContext *context, StringRef type) {
  return Base::get(context, RUST_TYPE, type);
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

RustType RustType::getTupleTy(RustDialect *dialect,
                              ArrayRef<RustType> elements) {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "(";
  for (unsigned i = 0; i < elements.size(); i++) {
    if (i != 0)
      s << ", ";
    s << elements[i].getRustType();
  }
  s << ")";
  return RustType::get(dialect->getContext(), s.str());
}
//===----------------------------------------------------------------------===//
// RustStructType
//===----------------------------------------------------------------------===//

struct RustStructTypeStorage : public TypeStorage {
  RustStructTypeStorage(ArrayRef<RustStructType::StructFieldTy> fields,
                        unsigned id)
      : structFields(fields.begin(), fields.end()), id(id) {}

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

private:
  static unsigned idCounter;
};

unsigned RustStructTypeStorage::idCounter = 0;

RustStructType RustStructType::get(RustDialect *dialect,
                                   ArrayRef<StructFieldTy> fields) {
  mlir::MLIRContext *ctx = fields.front().second.getContext();
  return Base::get(ctx, rust::types::RUST_STRUCT, fields);
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
std::string RustStructTypeStorage::getRustType() const {
  std::string str;
  llvm::raw_string_ostream s(str);

  s << "struct#" << id << "<";
  for (unsigned i = 0; i < structFields.size(); i++) {
    if (i != 0)
      s << ",";
    s << structFields[i].first.getValue() << ":" << structFields[i].second;
  }
  s << ">";

  return s.str();
}

RustPrinterStream &
RustStructTypeStorage::printAsRust(RustPrinterStream &ps) const {

  llvm::raw_ostream &os = ps.getNamedTypesStream();

  // First ensure that any structs used by this struct are defined
  for (unsigned i = 0; i < structFields.size(); i++)
    if (structFields[i].second.isa<RustStructType>())
      ps.writeStructDefiniton(structFields[i].second.cast<RustStructType>());

  os << "pub struct ";
  printAsRustNamedType(os) << " {\n  ";

  for (unsigned i = 0; i < structFields.size(); i++) {
    if (i != 0)
      os << ",\n  ";
    os << structFields[i].first.getValue() << " : ";
    Type t = structFields[i].second;
    if (t.isa<RustType>())
      t.cast<RustType>().printAsRust(os);
    else
      t.cast<RustStructType>().printAsRustNamedType(os);
  }
  os << "\n}\n";
  return ps;
}

raw_ostream &
RustStructTypeStorage::printAsRustNamedType(raw_ostream &os) const {

  os << "ArcStruct" << id;
  return os;
}

} // namespace types
} // namespace rust
