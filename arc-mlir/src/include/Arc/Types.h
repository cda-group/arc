//===- Dialect definition for the Arc IR ----------------------------------===//
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
// Defines the types of the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_TYPES_H_
#define ARC_TYPES_H_

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

using namespace mlir;

namespace arc {
namespace types {

//===----------------------------------------------------------------------===//
// Arc Type Functions
//===----------------------------------------------------------------------===//

bool isValueType(Type type);

//===----------------------------------------------------------------------===//
// Arc Type Storages
//===----------------------------------------------------------------------===//

struct SinkStreamTypeStorage;
struct SourceStreamTypeStorage;
struct StreamTypeBaseStorage;
struct StreamTypeStorage;
struct EnumTypeStorage;
struct StructTypeStorage;
struct ADTTypeStorage;
struct ADTGenericTypeStorage;

//===----------------------------------------------------------------------===//
// Arc Types
//===----------------------------------------------------------------------===//

class ADTType
    : public mlir::Type::TypeBase<ADTType, mlir::Type, ADTTypeStorage> {
public:
  using Base::Base;

  static ADTType get(mlir::MLIRContext *ctx, StringRef rustType);

  /// Returns the Rust name for this type.
  StringRef getTypeName() const;

  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};

class ADTGenericType : public mlir::Type::TypeBase<ADTGenericType, mlir::Type,
                                                   ADTGenericTypeStorage> {
public:
  using Base::Base;

  static ADTGenericType get(mlir::MLIRContext *ctx, StringRef name,
                            llvm::ArrayRef<mlir::Type> parameterTypes);

  StringRef getTemplateName() const;
  llvm::ArrayRef<mlir::Type> getParameterTypes() const;

  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};

class StreamTypeBase : public Type {
public:
  virtual ~StreamTypeBase(){};
  using ImplType = StreamTypeBaseStorage;
  using Type::Type;

  Type getElementType() const;
  Type getKeyType() const;
  StringRef getKeyword() const;
  virtual void print(DialectAsmPrinter &os) const;
  static Type parse(DialectAsmParser &parser);

protected:
  static Optional<std::pair<Type, Type>>
  parseStreamType(DialectAsmParser &parser);
};

class SinkStreamType
    : public mlir::Type::TypeBase<SinkStreamType, StreamTypeBase,
                                  SinkStreamTypeStorage> {
public:
  using Base::Base;

  static SinkStreamType get(mlir::Type keyType, mlir::Type elementType);

  static Type parse(DialectAsmParser &parser);
};

class SourceStreamType
    : public mlir::Type::TypeBase<SourceStreamType, StreamTypeBase,
                                  SourceStreamTypeStorage> {
public:
  using Base::Base;

  static SourceStreamType get(mlir::Type keyType, mlir::Type elementType);

  static Type parse(DialectAsmParser &parser);
};

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
public:
  using Base::Base;

  typedef std::pair<mlir::StringAttr, mlir::Type> FieldTy;

  static StructType get(mlir::MLIRContext *ctx, bool isCompact,
                        llvm::ArrayRef<FieldTy> elementTypes);

  /// Returns the fields of this struct type.
  llvm::ArrayRef<FieldTy> getFields() const;

  /// Returns the number of fields held by this struct.
  size_t getNumFields() const;

  // Returns true if this struct is compact
  bool isCompact() const;

  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};

class EnumType
    : public mlir::Type::TypeBase<EnumType, mlir::Type, EnumTypeStorage> {
public:
  using Base::Base;

  typedef std::pair<mlir::StringAttr, mlir::Type> VariantTy;

  static EnumType get(llvm::ArrayRef<VariantTy> elementTypes);

  /// Returns the variants of this enum type.
  llvm::ArrayRef<VariantTy> getVariants() const;

  /// Returns the number of variants in this enum.
  size_t getNumVariants() const;

  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};
} // namespace types
} // namespace arc

#endif // ARC_TYPES_H_
