//====- LowerToRust.cpp - Lowering from Arc+MLIR to Rust --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark task functions to use the nonpersistent Rust code generation
// strategy.
//
//===----------------------------------------------------------------------===//

#include "Arc/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace arc;

#define DEBUG_TYPE "to-nonpersistent"

namespace {
struct ToNonpersistent : public ToNonpersistentBase<ToNonpersistent> {
  void runOnOperation() final;
};
} // end anonymous namespace.

void ToNonpersistent::runOnOperation() {
  getOperation().walk([&](FuncOp f) {
    if (f->hasAttr("arc.is_task")) {
      f->setAttr("arc.is_toplevel_task_function", UnitAttr::get(&getContext()));
      f->setAttr("arc.use_nonpersistent", UnitAttr::get(&getContext()));
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createToNonpersistent() {
  return std::make_unique<ToNonpersistent>();
}
