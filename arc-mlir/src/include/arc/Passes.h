//===- Passes.h - Arc Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Arc.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_PASSES_H_
#define ARC_PASSES_H_

#include <memory>

namespace mlir {
class Pass;

namespace arc {
std::unique_ptr<Pass> createTypeInferencePass();
} // end namespace arc
} // end namespace mlir

#endif // ARC_PASSES_H_
