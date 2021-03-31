//===- Arcoptmain.cpp - ARC Mlit Optimizer Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that runs an optimization pass and prints the
// result back out. It is designed to support unit testing. Mainly
// stolen from MlirOptMain.cpp.
//
//===----------------------------------------------------------------------===//

#include "Arc/ArcOptMain.h"
#include "Rust/Rust.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;
using namespace llvm;
using namespace rust;
using llvm::SMLoc;
namespace cl = llvm::cl;

static cl::opt<bool> rustOutputInline("inline-rust",
                                      cl::desc("Produce an inline Rust module"),
                                      cl::value_desc("filename"));
/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult performActions(raw_ostream &os, bool verifyDiagnostics,
                                    bool verifyPasses, SourceMgr &sourceMgr,
                                    MLIRContext *context,
                                    const PassPipelineCLParser &passPipeline) {
  OwningModuleRef module(parseSourceFile(sourceMgr, context));
  if (!module)
    return failure();

  // Apply any pass manager command line options.
  PassManager pm(context, OpPassManager::Nesting::Implicit);
  pm.enableVerifier(verifyPasses);
  applyPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(context)) << msg;
    return failure();
  };

  // Build the provided pipeline.
  if (failed(passPipeline.addToPipeline(pm, errorHandler)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*module)))
    return failure();

  if (rustOutputInline) {
    if (failed(writeModuleAsInline(module.get(), os)))
      return failure();
    return success();
  }

  // Print the output.
  module->print(os);
  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult processBuffer(raw_ostream &os,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   bool verifyDiagnostics, bool verifyPasses,
                                   const PassPipelineCLParser &passPipeline,
                                   DialectRegistry &registry) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Parse the input file.
  MLIRContext context(registry);

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return performActions(os, verifyDiagnostics, verifyPasses, sourceMgr,
                          &context, passPipeline);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, verifyDiagnostics, verifyPasses, sourceMgr, &context,
                       passPipeline);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

LogicalResult mlir::ArcOptMain(raw_ostream &os,
                               std::unique_ptr<MemoryBuffer> buffer,
                               const PassPipelineCLParser &passPipeline,
                               DialectRegistry &registry, bool splitInputFile,
                               bool verifyDiagnostics, bool verifyPasses) {
  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  if (splitInputFile)
    return splitAndProcessBuffer(
        std::move(buffer),
        [&](std::unique_ptr<MemoryBuffer> chunkBuffer, raw_ostream &os) {
          return processBuffer(os, std::move(chunkBuffer), verifyDiagnostics,
                               verifyPasses, passPipeline, registry);
        },
        os);

  return processBuffer(os, std::move(buffer), verifyDiagnostics, verifyPasses,
                       passPipeline, registry);
}
