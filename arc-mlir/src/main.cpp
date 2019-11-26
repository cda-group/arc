/* Structure stolen from mlir-opt */
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/FileUtilities.h>

#include <memory>
#include <mlir/Analysis/Verifier.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/Parser.h>

namespace cl = llvm::cl;

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input arc file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename(cl::Positional,
                                           cl::desc("<output mlir file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

int main(int argc, char *argv[]) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "arc-mlir tool\n");

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << "Could not open input file: " << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << "Could not open output file: " << errorMessage << "\n";
    exit(1);
  }

  MLIRContext context;
  SourceMgr sourceMgr;

  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  OwningModuleRef module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  module->dump();

  return 0;
}
