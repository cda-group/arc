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
// This file implements the IR Dialect for the Arc language.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_DIALECT_H_
#define ARC_DIALECT_H_

#include "arc/Types.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Operation.h>
#include <mlir/Analysis/InferTypeOpInterface.h>

using namespace mlir;

namespace arc {

/// This is the definition of the Arc dialect.
class ArcDialect : public mlir::Dialect {
public:
  explicit ArcDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "arc"; }
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &os) const override;
};

/// Include the auto-generated header file containing the declarations of the
/// arc operations.
#define GET_OP_CLASSES
#include "arc/Ops.h.inc"

} // end namespace arc

#endif // ARC_DIALECT_H_
