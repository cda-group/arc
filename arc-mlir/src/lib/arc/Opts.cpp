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

#include "arc/Dialect.h"
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

struct ConstantFoldIf : public RewritePattern {
  ConstantFoldIf(MLIRContext *ctx) : RewritePattern("arc.if", 1, ctx) {}

  PatternMatchResult match(Operation *op) const override {
    Operation *def = op->getOperand(0).getDefiningOp();
    if (def && isa<ConstantOp>(def))
      return matchSuccess();
    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    int64_t cond =
        cast<ConstantIntOp>(op->getOperand(0).getDefiningOp()).getValue();
    Region &region = op->getRegion(cond ? 0 : 1);
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
  }
};

#include "Opts.inc"

} // end anonymous namespace

void MakeVectorOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *ctx) {
  populateWithGenerated(ctx, &results);
}

void IfOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                       MLIRContext *ctx) {
  results.insert<ConstantFoldIf>(ctx);
}
