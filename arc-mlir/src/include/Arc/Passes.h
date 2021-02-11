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

#include <mlir/Pass/Pass.h>

#include <memory>

using namespace mlir;

namespace arc {

void registerArcPasses();

/// Create a pass for lowering to operations in the `Rust` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createLowerToRustPass();

} // namespace arc

#endif // ARC_PASSES_H
