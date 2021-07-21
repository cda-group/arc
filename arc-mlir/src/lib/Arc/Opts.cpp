//===- Pattern Match Optimizations for ARC --------------------------------===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 KTH Royal Institute of Technology.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines the optimizations of the Arc dialect.
//
//===----------------------------------------------------------------------===//
#include <llvm/ADT/DenseMapInfo.h>

#include "Arc/Arc.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace llvm;
using namespace arc;

namespace {

bool AllValuesAreConstant(Operation::operand_range &ops) {
  for (const mlir::Value &a : ops) {
    Operation *op = a.getDefiningOp();
    if (!op || !isa<ConstantOp>(op)) // function arguments have no defining op
      return false;
  }
  return true;
}

DenseElementsAttr
ConstantValuesToDenseAttributes(mlir::OpResult result,
                                Operation::operand_range &ops) {
  ShapedType st = result.getType().cast<ShapedType>();
  std::vector<Attribute> attribs;
  for (const mlir::Value &a : ops) {
    ConstantOp def = cast<ConstantOp>(a.getDefiningOp());
    attribs.push_back(def.getValue());
  }
  return DenseElementsAttr::get(st, llvm::makeArrayRef(attribs));
}

struct ConstantFoldIf : public mlir::OpRewritePattern<arc::IfOp> {
  ConstantFoldIf(MLIRContext *ctx)
      : OpRewritePattern<arc::IfOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(arc::IfOp op, PatternRewriter &rewriter) const override {
    Operation *def = op.getOperand().getDefiningOp();

    if (!def || !isa<ConstantOp>(def))
      return failure();

    int64_t cond = cast<mlir::ConstantIntOp>(def).getValue();
    Region &region = cond ? op.thenRegion() : op.elseRegion();
    Block &block = region.getBlocks().front();

    Operation *block_result =
        block.getTerminator()->getOperand(0).getDefiningOp();
    Operation *cloned_result = nullptr;
    BlockAndValueMapping mapper;
    // We do the rewrite manually and not using
    // PatternRewriter::cloneRegion*() as we don't want to preserve
    // the blocks.
    for (auto &to_clone : block) {
      if (&to_clone == block.getTerminator()) {
        if (auto mappedOp = mapper.lookupOrNull(block_result->getResult(0))) {
          cloned_result = mappedOp.getDefiningOp();
        } else {
          cloned_result = block_result;
        }
        break;
      }
      Operation *cloned = to_clone.clone(mapper);
      rewriter.insert(cloned);
    }
    rewriter.replaceOp(op, cloned_result->getResults());
    return success();
  }
};

struct ConstantFoldIndexTuple
    : public mlir::OpRewritePattern<arc::IndexTupleOp> {
  ConstantFoldIndexTuple(MLIRContext *ctx)
      : OpRewritePattern<arc::IndexTupleOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(arc::IndexTupleOp op,
                  PatternRewriter &rewriter) const override {
    Operation *def = op.value().getDefiningOp();

    arc::MakeTupleOp mt = def ? dyn_cast<arc::MakeTupleOp>(def) : nullptr;
    if (!mt)
      return failure();
    rewriter.replaceOp(op, mt.values()[op.index()]);
    return success();
  }
};

struct ConstantFoldEnumAccess
    : public mlir::OpRewritePattern<arc::EnumAccessOp> {
  ConstantFoldEnumAccess(MLIRContext *ctx)
      : OpRewritePattern<arc::EnumAccessOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(arc::EnumAccessOp op,
                  PatternRewriter &rewriter) const override {
    Operation *def = op.value().getDefiningOp();
    arc::MakeEnumOp me = dyn_cast_or_null<arc::MakeEnumOp>(def);
    if (!me || !me.variant().equals(op.variant()))
      return failure();
    rewriter.replaceOp(op, me.values()[0]);
    return success();
  }
};

struct ConstantFoldStructAccess
    : public mlir::OpRewritePattern<arc::StructAccessOp> {
  ConstantFoldStructAccess(MLIRContext *ctx)
      : OpRewritePattern<arc::StructAccessOp>(ctx, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(arc::StructAccessOp op,
                  PatternRewriter &rewriter) const override {
    Operation *def = op.value().getDefiningOp();
    auto field = op.field();
    arc::MakeStructOp ms = dyn_cast_or_null<arc::MakeStructOp>(def);
    if (!ms)
      return failure();
    auto st = ms.getType().cast<arc::types::StructType>();
    unsigned idx = 0;
    for (auto &i : st.getFields()) {
      if (i.first.getValue().equals(field)) {
        rewriter.replaceOp(op, ms.values()[idx]);
        return success();
      }
      idx++;
    }
    return failure();
  }
};

#include "Arc/ArcOpts.h.inc"

} // end anonymous namespace

void EnumAccessOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *ctx) {
  results.insert<ConstantFoldEnumAccess>(ctx);
}

void MakeVectorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *ctx) {
  populateWithGenerated(results);
}

void IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *ctx) {
  results.insert<ConstantFoldIf>(ctx);
}

void IndexTupleOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *ctx) {
  results.insert<ConstantFoldIndexTuple>(ctx);
}

void StructAccessOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *ctx) {
  results.insert<ConstantFoldStructAccess>(ctx);
}
