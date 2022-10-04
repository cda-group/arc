//====- LowerToRust.cpp - Lowering from Arc+MLIR to Rust --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a conversion of arc-mlir functions containing
// blocking operations to FSMs. Boilerplate stolen from the MLIR Toy
// dialect.
//
//===----------------------------------------------------------------------===//

#include "Arc/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace arc;

#define DEBUG_TYPE "to-fsm"

//===----------------------------------------------------------------------===//
// ToFSMPass
//===----------------------------------------------------------------------===//

/// This is a lowering of arc operations to the Rust dialect.
namespace {

#define GEN_PASS_DEF_TOFSM
#include "Arc/Passes.h.inc"

struct ToFSMPass : public impl::ToFSMBase<ToFSMPass> {
  void runOnOperation() final;
};
} // end anonymous namespace.

class SplitAtStatepoint : public RewritePattern {
public:
  SplitAtStatepoint(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *block = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *restBlock = rewriter.splitBlock(block, opPosition);

    rewriter.setInsertionPointToEnd(block);
    Operation *br = rewriter.create<mlir::cf::BranchOp>(loc, restBlock);

    rewriter.startRootUpdate(op);
    br->setAttr("arc.br_to_statepoint", UnitAttr::get(getContext()));
    op->setAttr("arc.statepoint", UnitAttr::get(getContext()));
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

void ToFSMPass::runOnOperation() {
  SmallVector<mlir::func::FuncOp, 4> funs;
  getOperation().walk([&](mlir::func::FuncOp f) {
    if (f->hasAttr("arc.is_task"))
      funs.push_back(f);
  });

  for (mlir::func::FuncOp f : funs) {
    LLVM_DEBUG({
      llvm::errs() << "=== Function ===\n";
      llvm::errs() << f << "\n";
    });

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      // We trigger the rewrite pattern by considering receive
      // operations without an arc.statepoint attribute illegal.
      if (arc::ReceiveOp ro = dyn_cast<arc::ReceiveOp>(*op))
        return op->hasAttr("arc.statepoint");
      return true;
    });

    RewritePatternSet ps(&getContext());
    ps.add<SplitAtStatepoint>(1, ps.getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(ps))))
      signalPassFailure();

    LLVM_DEBUG({
      llvm::errs() << "=== After splits ===\n";
      llvm::errs() << f << "\n";
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createToFSMPass() {
  return std::make_unique<ToFSMPass>();
}
