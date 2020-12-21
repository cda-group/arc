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
bool isBuilderType(Type type);

//===----------------------------------------------------------------------===//
// Arc Type Storages
//===----------------------------------------------------------------------===//

struct ArconTypeStorage;
struct ArconValueTypeStorage;
struct ArconAppenderTypeStorage;
struct BuilderTypeStorage;
struct AppenderTypeStorage;
struct StreamTypeStorage;
struct StructTypeStorage;

//===----------------------------------------------------------------------===//
// Arc Types
//===----------------------------------------------------------------------===//

class BuilderType : public Type {
public:
  using ImplType = BuilderTypeStorage;
  using Type::Type;

  Type getMergeType() const;
  Type getResultType() const;
};

class AppenderType
    : public Type::TypeBase<AppenderType, BuilderType, AppenderTypeStorage> {
public:
  using Base::Base;

  static AppenderType get(Type mergeType, RankedTensorType resultType);
  static AppenderType getChecked(Type mergeType, RankedTensorType resultType,
                                 Location loc);
  static LogicalResult
  verifyConstructionInvariants(Location loc, Type mergeType,
                               RankedTensorType resultType);
  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};

class ArconType : public Type {
public:
  virtual ~ArconType(){};
  using ImplType = ArconTypeStorage;
  using Type::Type;

  Type getContainedType() const;
  StringRef getKeyword() const;
  virtual void print(DialectAsmPrinter &os) const;
};

class ArconValueType : public mlir::Type::TypeBase<ArconValueType, ArconType,
                                                   ArconValueTypeStorage> {
public:
  using Base::Base;

  static ArconValueType get(mlir::Type elementType);

  /// Returns the type of the stream elements
  mlir::Type getType() const;

  static Type parse(DialectAsmParser &parser);
};

class ArconAppenderType
    : public mlir::Type::TypeBase<ArconAppenderType, ArconType,
                                  ArconAppenderTypeStorage> {
public:
  using Base::Base;

  static ArconAppenderType get(mlir::Type elementType);

  /// Returns the type of the stream elements
  mlir::Type getType() const;

  static Type parse(DialectAsmParser &parser);
};

class StreamType
    : public mlir::Type::TypeBase<StreamType, ArconType, StreamTypeStorage> {
public:
  using Base::Base;

  static StreamType get(mlir::Type elementType);

  /// Returns the type of the stream elements
  mlir::Type getType() const;

  static Type parse(DialectAsmParser &parser);
};

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
public:
  using Base::Base;

  typedef std::pair<mlir::StringAttr, mlir::Type> FieldTy;

  static StructType get(llvm::ArrayRef<FieldTy> elementTypes);

  /// Returns the fields of this struct type.
  llvm::ArrayRef<FieldTy> getFields() const;

  /// Returns the number of fields held by this struct.
  size_t getNumFields() const;

  static Type parse(DialectAsmParser &parser);
  void print(DialectAsmPrinter &os) const;
};
} // namespace types
} // namespace arc

#endif // ARC_TYPES_H_
