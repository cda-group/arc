//====- RemoveSCF.cpp - Remove structured control flow --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a conversion of arc-mlir functions containing
// structured control flow to plain branches between blocks.
//
// Much content stolen from the SCFToStandard conversion.
//
//===----------------------------------------------------------------------===//
#include "Arc/Arc.h"
#include "Arc/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::cf;
using namespace arc;
using namespace llvm;

//===----------------------------------------------------------------------===//
// RemoveSCFPass
//===----------------------------------------------------------------------===//

cl::opt<bool> OnlyRunOnTasks("remove-scf-only-tasks",
                             cl::desc("Only run the pass on tasks"),
                             cl::init(false));

/// This is a lowering of arc operations to the Rust dialect.
namespace {
struct RemoveSCF : public RemoveSCFBase<RemoveSCF> {
  void runOnOperation() final;
};

struct IfLowering : public OpRewritePattern<arc::IfOp> {
  using OpRewritePattern<arc::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arc::IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

struct ArcReturnLowering : public OpRewritePattern<ArcReturnOp> {
  using OpRewritePattern<ArcReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ArcReturnOp ifOp,
                                PatternRewriter &rewriter) const override;
};

struct WhileLowering : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};

} // end anonymous namespace.

// Stolen from upstream's SCFToStandard but modified to handle
// arc.return.
LogicalResult IfLowering::matchAndRewrite(arc::IfOp ifOp,
                                          PatternRewriter &rewriter) const {
  auto loc = ifOp.getLoc();

  // Start by splitting the block containing the 'scf.if' into two parts.
  // The part before will contain the condition, the part after will be the
  // continuation point.
  auto *condBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
  Block *continueBlock;
  if (ifOp.getNumResults() == 0) {
    continueBlock = remainingOpsBlock;
  } else {
    continueBlock = rewriter.createBlock(
        remainingOpsBlock, ifOp.getResultTypes(),
        SmallVector<Location>(ifOp.getResultTypes().size(), loc));
    rewriter.create<BranchOp>(loc, remainingOpsBlock);
  }

  // Move blocks from the "then" region to the region containing 'scf.if',
  // place it before the continuation block, and branch to it.
  auto &thenRegion = ifOp.thenRegion();
  auto *thenBlock = &thenRegion.front();
  Operation *thenTerminator = thenRegion.back().getTerminator();
  ValueRange thenTerminatorOperands = thenTerminator->getOperands();
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  if (isa<arc::LoopBreakOp>(thenTerminator)) {
    // Leave it in, the while lowering will remove it
  } else if (isa<arc::ArcReturnOp>(thenTerminator) ||
             isa<func::ReturnOp>(thenTerminator)) {
    rewriter.create<func::ReturnOp>(loc, thenTerminator->getResultTypes(),
                                    thenTerminatorOperands);
    rewriter.eraseOp(thenTerminator);
  } else {
    rewriter.create<BranchOp>(loc, continueBlock, thenTerminatorOperands);
    rewriter.eraseOp(thenTerminator);
  }
  rewriter.inlineRegionBefore(thenRegion, continueBlock);

  // Move blocks from the "else" region (if present) to the region containing
  // 'scf.if', place it before the continuation block and branch to it.  It
  // will be placed after the "then" regions.
  auto *elseBlock = continueBlock;
  auto &elseRegion = ifOp.elseRegion();
  if (!elseRegion.empty()) {
    elseBlock = &elseRegion.front();
    Operation *elseTerminator = elseRegion.back().getTerminator();
    ValueRange elseTerminatorOperands = elseTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    if (isa<arc::LoopBreakOp>(thenTerminator)) {
      // Leave it in, the while lowering will remove it
    }
    if (isa<arc::ArcReturnOp>(elseTerminator) ||
        isa<func::ReturnOp>(elseTerminator)) {
      rewriter.create<func::ReturnOp>(loc, elseTerminator->getResultTypes(),
                                      elseTerminatorOperands);
      rewriter.eraseOp(elseTerminator);
    } else {
      rewriter.create<BranchOp>(loc, continueBlock, elseTerminatorOperands);
      rewriter.eraseOp(elseTerminator);
    }
    rewriter.inlineRegionBefore(elseRegion, continueBlock);
  }
  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<CondBranchOp>(loc, ifOp.condition(), thenBlock,
                                /*trueArgs=*/ArrayRef<Value>(), elseBlock,
                                /*falseArgs=*/ArrayRef<Value>());
  // Ok, we're done!
  rewriter.replaceOp(ifOp, continueBlock->getArguments());
  return success();
}

LogicalResult
ArcReturnLowering::matchAndRewrite(ArcReturnOp op,
                                   PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.operands());
  return success();
}

// Stolen from upstream's SCFToStandard but modified to handle
// arc.return and arc.loop.break.
LogicalResult WhileLowering::matchAndRewrite(scf::WhileOp whileOp,
                                             PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = whileOp.getLoc();

  // Split the current block before the WhileOp to create the inlining point.
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  if (whileOp.getNumResults() != 0) {
    Block *t = continuation;
    auto resultTypes = whileOp.getResultTypes();
    continuation = rewriter.createBlock(
        t, resultTypes, SmallVector<Location>(resultTypes.size(), loc));
    rewriter.create<BranchOp>(loc, t);
  }

  // Before we start to move operations from the enclosed regions, we
  // look up all arc.loop.break-operations which breaks out of the
  // current loop.
  std::vector<arc::LoopBreakOp> breaks;
  whileOp.walk([&](arc::LoopBreakOp op) {
    if (op->getAttr("arc.loop_id") == whileOp->getAttr("arc.loop_id"))
      breaks.push_back(op);
  });

  // Inline both regions.
  Block *after = &whileOp.getAfter().front();
  Block *afterLast = &whileOp.getAfter().back();
  Block *before = &whileOp.getBefore().front();
  Block *beforeLast = &whileOp.getBefore().back();
  rewriter.inlineRegionBefore(whileOp.getAfter(), continuation);
  rewriter.inlineRegionBefore(whileOp.getBefore(), after);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<BranchOp>(loc, before, whileOp.getInits());

  // Replace terminators with branches.
  rewriter.setInsertionPointToEnd(beforeLast);
  auto condOp = cast<scf::ConditionOp>(beforeLast->getTerminator());
  rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, condOp.getCondition(),
                                            after, condOp.getArgs(),
                                            continuation, condOp.getArgs());

  // Patch away the arc.loop.break-operations with direct branches
  for (arc::LoopBreakOp op : breaks) {
    rewriter.setInsertionPointAfter(op);
    rewriter.replaceOpWithNewOp<BranchOp>(op, continuation, op.results());
  }

  rewriter.setInsertionPointToEnd(afterLast);
  auto yieldOp = cast<scf::YieldOp>(afterLast->getTerminator());
  rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, before, yieldOp.getResults());

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(whileOp, condOp.getArgs());

  return success();
}

void RemoveSCF::runOnOperation() {
  mlir::func::FuncOp f = getOperation();

  if (OnlyRunOnTasks && !f->hasAttr("arc.is_task"))
    return;

  // In order to match arc.loop.breaks to their enclosing scf.while we
  // add an unique attribute to each scf.while. We then tag all
  // arc.loop.breaks with the id of their closest enclosing scf.while.
  unsigned loop_id = 0;
  f.walk([&](scf::WhileOp op) {
    op->setAttr("arc.loop_id",
                IntegerAttr::get(IndexType::get(&getContext()), loop_id++));
  });
  f.walk([&](arc::LoopBreakOp op) {
    scf::WhileOp whileOp = op->getParentOfType<scf::WhileOp>();
    op->setAttr("arc.loop_id", whileOp->getAttr("arc.loop_id"));
  });

  // Lower arc.if and arc.return operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<arc::IfOp>();
  target.addIllegalOp<ArcReturnOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet ifPattern(&getContext());
  ifPattern.add<IfLowering, ArcReturnLowering>(ifPattern.getContext());
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(ifPattern))))
    signalPassFailure();

  // Lower scf.while operations.
  target.addIllegalOp<scf::WhileOp>();
  RewritePatternSet whilePattern(&getContext());
  whilePattern.add<WhileLowering>(ifPattern.getContext());
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(whilePattern))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<mlir::func::FuncOp>> arc::createRemoveSCFPass() {
  return std::make_unique<RemoveSCF>();
}
