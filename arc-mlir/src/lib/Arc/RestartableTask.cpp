//====- LowerToRust.cpp - Lowering from Arc+MLIR to Rust --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Split task functions which have gone through to-fsm and to-scf into
// two functions. The original function creates the initial FSM state
// and returns it. The second function takes the FSM state and
// executes it.
//
//===----------------------------------------------------------------------===//

#include "Arc/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace arc;

#define DEBUG_TYPE "restartable-task"

//===----------------------------------------------------------------------===//
// RestartableTaskPass
//===----------------------------------------------------------------------===//

/// This is a lowering of arc operations to the Rust dialect.
namespace {

#define GEN_PASS_DEF_RESTARTABLETASK
#include "Arc/Passes.h.inc"

struct RestartableTaskPass
    : public impl::RestartableTaskBase<RestartableTaskPass> {
  void runOnOperation() final;
};
} // end anonymous namespace.

void RestartableTaskPass::runOnOperation() {
  // Find all task functions
  SmallVector<mlir::func::FuncOp, 4> funs;
  getOperation().walk([&](mlir::func::FuncOp f) {
    if (f->hasAttr("arc.is_task"))
      funs.push_back(f);
  });

  for (mlir::func::FuncOp f : funs) {
    arc::MakeStructOp makeStruct;
    arc::MakeEnumOp makeEnum;
    scf::WhileOp whileOp;

    SmallVector<Operation *, 4> toClone;

    // Iterate over the ops in the entry block to find the
    // make_struct, the construction of the initial enum and the
    // while-loop forming the task loop. While doing this we remember
    // any hoisted operations which will be cloned into to body
    // function.
    for (Operation &o : f->getRegion(0).front().getOperations()) {
      if (o.hasAttr("arc.task_initial_struct")) {
        makeStruct = cast<arc::MakeStructOp>(o);
      } else if (o.hasAttr("arc.task_initial_enum")) {
        makeEnum = cast<arc::MakeEnumOp>(o);
      } else if (o.hasAttr("arc.task_loop")) {
        whileOp = cast<scf::WhileOp>(o);
        break;
      } else {
        toClone.push_back(&o);
      }
    }
    // Create the FSM body function
    FunctionType funTy =
        FunctionType::get(&getContext(), makeEnum.getResult().getType(), {});
    std::string name =
        f->getAttrOfType<StringAttr>("sym_name").getValue().str() + "_body";
    func::FuncOp bodyFunc = func::FuncOp::create(f->getLoc(), name, funTy);
    bodyFunc.setPrivate();
    // Only mark the body function as a task
    bodyFunc->setAttr("arc.is_task", UnitAttr::get(&getContext()));
    f->removeAttr("arc.is_task");
    f->setAttr("rust.declare", UnitAttr::get(&getContext()));

    // Move asynch attribute to the body
    if (f->hasAttr("rust.async")) {
      f->removeAttr("rust.async");
      bodyFunc->setAttr("rust.async", UnitAttr::get(&getContext()));
    }
    // Move annotation_task_body attribute to the body as rust.annotation
    if (f->hasAttr("rust.annotation_task_body")) {
      bodyFunc->setAttr("rust.annotation",
                        f->getAttr("rust.annotation_task_body"));
      f->removeAttr("rust.annotation_task_body");
    }

    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    symbolTable.insert(bodyFunc);

    LLVM_DEBUG({
      llvm::dbgs() << "=== Function: ===\n";
      llvm::dbgs() << "struct: " << makeStruct << "\n";
      llvm::dbgs() << "enum: " << makeEnum << "\n";
      llvm::dbgs() << "fun-type: " << funTy << "\n";
      for (Operation *op : toClone)
        llvm::dbgs() << "Will clone: " << *op << "\n";
    });

    // Clone the while-loop into the new body function
    IRMapping map;
    IRRewriter rewriter(&getContext());
    Block *entryBB = rewriter.createBlock(&bodyFunc->getRegion(0));
    auto blockArgs = SmallVector<Location>(1, f->getLoc());
    entryBB->addArguments({makeEnum.getResult().getType()}, blockArgs);
    rewriter.setInsertionPointToEnd(entryBB);

    map.map(makeEnum.getResult(), entryBB->getArgument(0));
    for (Operation *op : toClone) {
      Operation *newOp = rewriter.clone(*op, map);
      if (op->getNumResults())
        map.map(op->getResult(0), newOp->getResult(0));
    }
    rewriter.clone(*whileOp, map);
    rewriter.setInsertionPointToEnd(entryBB);
    rewriter.create<func::ReturnOp>(f->getLoc());

    // Patch the original function and replace the while-loop with
    // a call to the new function.
    rewriter.setInsertionPoint(whileOp);
    auto callArgs = SmallVector<Value, 1>{makeEnum.getResult()};
    rewriter.eraseOp(whileOp->getBlock()->getTerminator());
    rewriter.create<func::ReturnOp>(makeEnum.getLoc(), callArgs);
    rewriter.eraseOp(whileOp);

    // Patch the return type
    FunctionType origFunTy = f.getFunctionType();
    FunctionType modFunTy = FunctionType::get(
        &getContext(), origFunTy.getInputs(), {makeEnum.getResult().getType()});
    f.setFunctionTypeAttr(TypeAttr::get(modFunTy));

    LLVM_DEBUG({
      llvm::dbgs() << "=== the modified task funktion ===\n";
      f->dump();
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createRestartableTaskPass() {
  return std::make_unique<RestartableTaskPass>();
}
