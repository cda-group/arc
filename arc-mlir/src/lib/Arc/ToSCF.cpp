//====- ToSCF.cpp - urn unstructured control flow into SCF --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a conversion of arc-mlir functions built with
// branches between blocks to structured control flow.
//
//
//===----------------------------------------------------------------------===//
#include "Arc/Arc.h"
#include "Arc/Passes.h"
#include "Arc/Types.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;
using namespace mlir::cf;
using namespace arc;

#define DEBUG_TYPE "to-scf"

//===----------------------------------------------------------------------===//
// ToSCFPass
//===----------------------------------------------------------------------===//

/// This is a lowering of arc operations to the Rust dialect.
namespace {
struct ToSCF : public ToSCFBase<ToSCF> {
  void runOnOperation() final;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
};

} // end anonymous namespace.

struct FunPattern : public OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::func::FuncOp fun,
                                PatternRewriter &rewriter) const override {
    bool foundBr = false;
    fun->walk<WalkOrder::PreOrder>([&](BranchOp br) { foundBr = true; });
    fun->walk<WalkOrder::PreOrder>([&](CondBranchOp br) { foundBr = true; });

    if (!foundBr) {
      LLVM_DEBUG(llvm::dbgs()
                     << "No unstructured control flow instructions found in "
                     << fun.getName() << "\n";);
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
                   << "Unstructured control flow instructions found in "
                   << fun.getName() << "\n";);
    return transformFun(fun, rewriter);
  }

private:
  StringAttr getBlockArgFieldName(unsigned blockId, unsigned n) const {
    return StringAttr::get(getContext(), "Block" + std::to_string(blockId) +
                                             "Arg" + std::to_string(n));
  }

  StringAttr getBlockLiveFieldName(unsigned blockId, unsigned n) const {
    return StringAttr::get(getContext(), "Block" + std::to_string(blockId) +
                                             "Live" + std::to_string(n));
  }

  StringAttr getBlockVariantName(unsigned blockId) const {
    return StringAttr::get(getContext(),
                           "Block" + std::to_string(blockId) + "variant");
  }

  void
  buildBranchTo(Block *dest, ::mlir::Operation::operand_range destOps,
                DenseMap<Block *, arc::types::StructType> &block2variantStruct,
                DenseMap<Block *, StringAttr> &block2variantName,
                DenseMap<Block *, SmallVector<std::pair<Value, StringAttr>>>
                    &block2live2Field,
                types::EnumType stateTy, BlockAndValueMapping &map,
                Location loc, PatternRewriter &rewriter) const {
    SmallVector<Value, 4> destArgs;
    for (auto a : destOps)
      destArgs.push_back(map.lookup(a));
    for (auto l : block2live2Field[dest])
      destArgs.push_back(map.lookup(l.first));
    Value destStruct = rewriter.create<arc::MakeStructOp>(
        loc, block2variantStruct[dest], destArgs);
    Value destState = rewriter.create<MakeEnumOp>(loc, stateTy, destStruct,
                                                  block2variantName[dest]);
    rewriter.create<arc::ArcBlockResultOp>(loc, destState);
  }

  LogicalResult transformFun(mlir::func::FuncOp f,
                             PatternRewriter &rewriter) const {
    LLVM_DEBUG({
      llvm::dbgs() << "=== Function ===\n";
      auto &block = f->getRegion(0).front();
      llvm::dbgs() << "The entry block has " << block.getNumArguments()
                   << " arguments\n";

      llvm::dbgs() << f << "\n";
    });

    DenseMap<unsigned, Block *> id2block;
    DenseMap<Block *, unsigned> block2id;
    DenseMap<Block *, types::EnumType::VariantTy> block2variant;
    DenseMap<Block *, arc::types::StructType> block2variantStruct;
    DenseMap<Block *, StringAttr> block2variantName;
    DenseMap<Block *, SmallVector<std::pair<Value, StringAttr>>>
        block2live2Field;
    unsigned id = 0;

    f->walk<WalkOrder::PreOrder>([&](Block *block) {
      id2block[id] = block;
      block2id[block] = id++;
    });

    Liveness live(f);
    SmallVector<types::EnumType::VariantTy, 16> enumVariants;

    for (unsigned i = 0; i < id2block.size(); i++) {
      Block *b = id2block[i];
      SmallVector<types::StructType::FieldTy, 8> elements;

      LLVM_DEBUG({
        llvm::dbgs() << "** Block " << i << " **\n";
        b->dump();
        llvm::dbgs() << "Live in:\n";
      });
      const Liveness::ValueSetT &liveIn = live.getLiveIn(b);
      unsigned blockArgId = 0;
      for (auto a : b->getArgumentTypes()) {
        types::StructType::FieldTy elementType{
            getBlockArgFieldName(i, blockArgId++), a};
        elements.push_back(elementType);
      }
      unsigned liveId = 0;
      for (auto v : liveIn) {
        LLVM_DEBUG(llvm::dbgs() << "  " << v << "\n");
        StringAttr f = getBlockLiveFieldName(i, liveId++);
        types::StructType::FieldTy elementType{f, v.getType()};
        block2live2Field[b].push_back(std::make_pair(v, f));
        elements.push_back(elementType);
      }
      LLVM_DEBUG(llvm::dbgs() << "Live out\n");
      const Liveness::ValueSetT &liveOut = live.getLiveOut(b);
      for (auto v : liveOut)
        LLVM_DEBUG(llvm::dbgs() << "  " << v << "\n");
      // We need a struct for each block
      types::StructType t =
          types::StructType::get(getContext(), false, elements);
      block2variantStruct[b] = t;
      LLVM_DEBUG(llvm::dbgs() << "Entry struct type " << t << "\n");
      StringAttr variantName = getBlockVariantName(i);
      types::EnumType::VariantTy variant{variantName, t};
      block2variantName[b] = variantName;
      block2variant[b] = variant;
      enumVariants.push_back(variant);
    }

    // Create a variant to hold the result
    StringAttr returnValueVariantName =
        StringAttr::get(getContext(), "ReturnValue");
    FunctionType funTy = f.getFunctionType();
    Type funReturnTy =
        funTy.getNumResults() ? funTy.getResult(0) : rewriter.getNoneType();
    types::EnumType::VariantTy returnValueVariant{returnValueVariantName,
                                                  funReturnTy};
    enumVariants.push_back(returnValueVariant);

    // Create a new basic block to hold the loop
    auto &entryBB = f->getRegion(0).front();
    Block *newEntryBB = rewriter.createBlock(
        &entryBB, entryBB.getArgumentTypes(),
        SmallVector<Location>(entryBB.getArgumentTypes().size(), f->getLoc()));
    rewriter.setInsertionPointToEnd(newEntryBB);

    // Create the type for the state, it consists of the variants for
    // each block and the result.
    types::EnumType stateTy = types::EnumType::get(enumVariants);

    LLVM_DEBUG(llvm::dbgs() << "** Enum **\n" << stateTy << "\n");

    // Create the initial state for the loop
    Value initStruct = rewriter.create<arc::MakeStructOp>(
        f->getLoc(), block2variantStruct[&entryBB], newEntryBB->getArguments());
    Value initState = rewriter.create<MakeEnumOp>(
        f->getLoc(), stateTy, initStruct, block2variantName[&entryBB]);
    SmallVector<Type, 1> loopVarTypes;
    SmallVector<Value, 1> loopOps;

    loopVarTypes.push_back(stateTy);
    loopOps.push_back(initState);

    // Create the while loop and the before and after basic blocks
    scf::WhileOp loop =
        rewriter.create<scf::WhileOp>(f->getLoc(), loopVarTypes, loopOps);
    Block *beforeBB = rewriter.createBlock(&loop.getBefore());

    auto tmp = SmallVector<Location>(loopVarTypes.size(), f->getLoc());
    beforeBB->addArguments(loopVarTypes, tmp);
    Block *afterBB = rewriter.createBlock(&loop.getAfter());
    afterBB->addArguments(
        loopVarTypes, SmallVector<Location>(loopVarTypes.size(), f->getLoc()));

    // Fill in the before block
    rewriter.setInsertionPointToEnd(beforeBB);
    SmallVector<Type, 4> condTypes;
    SmallVector<Value, 4> condArgs;

    // Create the loop condition, it checks if we have the return
    // variant of the state enum.
    Value isDone = rewriter.create<arc::EnumCheckOp>(
        f->getLoc(), rewriter.getI1Type(), beforeBB->getArgument(0),
        returnValueVariantName);
    isDone = rewriter.create<arith::XOrIOp>(
        f->getLoc(), isDone,
        rewriter.create<arith::ConstantIntOp>(f->getLoc(), 1, 1));

    rewriter.create<scf::ConditionOp>(f->getLoc(), isDone,
                                      beforeBB->getArguments());

    Block *nextBB = afterBB;
    Block *bodyBB = afterBB;
    Operation *bodyResult = nullptr;
    // In the after block, we sequentially test the enum for the
    // different states.
    for (unsigned i = 0; i < id2block.size(); i++) {
      Block *b = id2block[i];
      rewriter.setInsertionPointToEnd(nextBB);

      if (i < id2block.size() - 1) {
        Value isThisBB = rewriter.create<arc::EnumCheckOp>(
            f->getLoc(), rewriter.getI1Type(), afterBB->getArgument(0),
            block2variantName[b]);

        IfOp thisIf =
            rewriter.create<arc::IfOp>(f->getLoc(), stateTy, isThisBB);
        if (i == 0) {
          bodyResult = thisIf;
        } else {
          rewriter.create<arc::ArcBlockResultOp>(f->getLoc(),
                                                 thisIf.getResult(0));
        }
        bodyBB = rewriter.createBlock(&thisIf.thenRegion());
        nextBB = rewriter.createBlock(&thisIf.elseRegion());
      } else {
        bodyBB = nextBB;
      }
      rewriter.setInsertionPointToEnd(bodyBB);

      // Prepare to clone the body by setting up a mapping which maps
      // the original values to values extracted from the state
      // struct.
      BlockAndValueMapping map;

      // First extract the struct for this BB
      types::EnumType::VariantTy variantInfo = block2variant[b];
      Value bbState = rewriter.create<arc::EnumAccessOp>(
          f.getLoc(), variantInfo.second, afterBB->getArgument(0),
          variantInfo.first);

      // Extract a value for each block argument from the state struct
      unsigned blockArgId = 0;
      for (BlockArgument &a : b->getArguments()) {
        Value field = rewriter.create<arc::StructAccessOp>(
            f.getLoc(), a.getType(), bbState,
            getBlockArgFieldName(block2id[b], blockArgId++));
        map.map(a, field);
      }

      // Extract a value for each live variable from the state struct
      for (auto fieldInfo : block2live2Field[b]) {
        Value field = rewriter.create<arc::StructAccessOp>(
            f.getLoc(), fieldInfo.first.getType(), bbState, fieldInfo.second);
        map.map(fieldInfo.first, field);
      }

      // Fill in body, we clone instructions one-by-one as the
      // rewriter only has methods for cloning regions, not blocks.
      for (Operation &op : b->getOperations()) {
        if (BranchOp br = dyn_cast<BranchOp>(op)) {
          Block *dest = br.getDest();
          buildBranchTo(dest, br.getDestOperands(), block2variantStruct,
                        block2variantName, block2live2Field, stateTy, map,
                        f->getLoc(), rewriter);
        } else if (CondBranchOp br = dyn_cast<CondBranchOp>(op)) {
          Value conditional = map.lookup(br.getCondition());
          IfOp destIf =
              rewriter.create<arc::IfOp>(f->getLoc(), stateTy, conditional);
          rewriter.create<arc::ArcBlockResultOp>(f->getLoc(),
                                                 destIf.getResult(0));

          Block *thenBB = rewriter.createBlock(&destIf.thenRegion());
          rewriter.setInsertionPointToEnd(thenBB);
          buildBranchTo(br.getTrueDest(), br.getTrueDestOperands(),
                        block2variantStruct, block2variantName,
                        block2live2Field, stateTy, map, f->getLoc(), rewriter);
          Block *elseBB = rewriter.createBlock(&destIf.elseRegion());
          rewriter.setInsertionPointToEnd(elseBB);
          buildBranchTo(br.getFalseDest(), br.getFalseDestOperands(),
                        block2variantStruct, block2variantName,
                        block2live2Field, stateTy, map, f->getLoc(), rewriter);
        } else if (func::ReturnOp r = dyn_cast<func::ReturnOp>(op)) {
          SmallVector<Value, 1> values;

          if (r.getNumOperands())
            values.push_back(map.lookup(r.operands()[0]));
          Value returnState = rewriter.create<MakeEnumOp>(
              f->getLoc(), stateTy, values, returnValueVariant.first);
          rewriter.create<arc::ArcBlockResultOp>(f->getLoc(), returnState);
        } else
          rewriter.clone(op, map);
      }
    }
    rewriter.setInsertionPointAfter(bodyResult);
    rewriter.create<scf::YieldOp>(f->getLoc(), bodyResult->getResult(0));

    // Return the result
    rewriter.setInsertionPointAfter(loop);
    if (f.getFunctionType().getNumResults()) {
      Value r = rewriter.create<arc::EnumAccessOp>(
          f.getLoc(), funReturnTy, loop.getResult(0), returnValueVariantName);
      rewriter.create<func::ReturnOp>(f.getLoc(), r);
    } else
      rewriter.create<func::ReturnOp>(f.getLoc());

    // Remove all basic blocks from the original function, only
    // leaving the new entry block.
    for (auto &i : block2id)
      i.first->dropAllReferences();
    for (auto &i : block2id)
      rewriter.eraseBlock(i.first);

    LLVM_DEBUG({
      llvm::dbgs() << "**** Result ****\n";
      f.dump();
    });
    return success();
  }
};

void ToSCF::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FunPattern>(patterns.getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createToSCFPass() {
  return std::make_unique<ToSCF>();
}
