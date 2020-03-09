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

#ifndef ARC_OPT_MAIN_H_
#define ARC_OPT_MAIN_H_

#include <memory>
#include <vector>

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
                         bool splitInputFile, bool verifyDiagnostics,
                         bool verifyPasses);

} // end namespace mlir

#endif // ARC_OPT_MAIN_H_
