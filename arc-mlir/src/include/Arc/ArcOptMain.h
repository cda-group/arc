//===- MlirOptMain.h - MLIR Optimizer Driver main ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for arc-mlir for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <vector>

#include "mlir/IR/Dialect.h"

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // end namespace llvm

namespace mlir {
struct LogicalResult;
class PassPipelineCLParser;

LogicalResult ArcOptMain(llvm::raw_ostream &os,
                         std::unique_ptr<llvm::MemoryBuffer> buffer,
                         const PassPipelineCLParser &passPipeline,
                         DialectRegistry &registry, bool splitInputFile,
                         bool verifyDiagnostics, bool verifyPasses);

} // end namespace mlir
