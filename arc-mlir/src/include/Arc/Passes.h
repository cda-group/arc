//===- Passes.h - Arc Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for
// Arc. Boilerplate stolen from the MLIR Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_PASSES_H
#define ARC_PASSES_H

#include <memory>
#include <mlir/Pass/Pass.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "Arc/Arc.h"
#include "Rust/Rust.h"

using namespace mlir;

namespace arc {

void registerArcPasses();

/// Create a pass for lowering to operations in the `Rust` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createLowerToRustPass();

/// Create a pass for converting structured flow-control to branches
/// between basic blocks.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createRemoveSCFPass();

/// Create a pass for converting branches
/// between basic blocks to structured flow-control.
std::unique_ptr<OperationPass<ModuleOp>> createToSCFPass();

/// Create a pass for converting blocking code to FSMs.
std::unique_ptr<OperationPass<ModuleOp>> createToFSMPass();

/// Create a pass for making a task restartable.
std::unique_ptr<OperationPass<ModuleOp>> createRestartableTaskPass();

} // namespace arc

#endif // ARC_PASSES_H
