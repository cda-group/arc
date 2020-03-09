//===- TypeInferencePass.cpp - Type Inference -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array types through function specialization.
//
//===----------------------------------------------------------------------===//

#include "arc/Dialect.h"
#include "arc/Passes.h"
#include "arc/TypeInferenceOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "type-inference"

using namespace mlir;
using namespace arc;

/// Include the auto-generated definitions for the type inference interfaces.
#include "arc/TypeInferenceOpInterface.cpp.inc"

namespace {
/// The TypeInferencePass is a FunctionPass that performs intra-procedural
/// type inference.
///
///    Algorithm:
///
///   1) Build a set containing all the operations that return a unknown type:
//       these are the operations that need type inference.
///   2) Iterate on the set:
///     a) find an operation to process: the next ready operation in the
///        set has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the set,
///     d) infer the type of its output from the argument types.
///   3) If the set is empty, the algorithm succeeded.
///
class TypeInferencePass : public mlir::FunctionPass<TypeInferencePass> {
public:
  void runOnFunction() override {
    auto f = getFunction();

    // Populate the set with the operations that need type inference:
    // these are operations that return a dynamic type.
    llvm::SmallPtrSet<mlir::Operation *, 16> opSet;
    f.walk([&](mlir::Operation *op) {
      if (returnsUnknownType(op))
        opSet.insert(op);
    });

    // Iterate on the operations in the set until all operations have been
    // inferred or no change happened (fix point).
    while (!opSet.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opSet, returnsUnknownType);
      if (nextop == opSet.end())
        break;

      Operation *op = *nextop;
      opSet.erase(op);

      // Ask the operation to infer its output types.
      LLVM_DEBUG(llvm::dbgs() << "Inferring type for: " << *op << "\n");
      if (auto typeOp = dyn_cast<TypeInference>(op)) {
        typeOp.inferTypes();
      } else {
        op->emitError("unable to infer type of operation without type "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation set isn't empty, this indicates a failure.
    if (!opSet.empty()) {
      f.emitError("Type inference failed, ")
          << opSet.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has a dynamically
  /// typed result.
  static bool returnsUnknownType(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !resultType.isa<UnknownType>();
    });
  }
};
} // end anonymous namespace

/// Create a Type Inference pass.
std::unique_ptr<mlir::Pass> mlir::arc::createTypeInferencePass() {
  return std::make_unique<TypeInferencePass>();
}
