//===- Pattern Match Optimizations for ARC --------------------------===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 RISE AB.
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
//===----------------------------------------------------------------------===//

#include "arc/arc-dialect.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace llvm;
using namespace arc;

namespace {

bool AllValuesAreConstant(Operation::operand_range &ops) {
  for (mlir::Value *a : ops) {
    Operation *op = a->getDefiningOp();
    if (!op || !isa<ConstantOp>(op)) // function arguments have no defining op
      return false;
  }
  return true;
}

DenseElementsAttr ToDenseAttribs(Value *result, Operation::operand_range &ops) {
  ShapedType st = result->getType().cast<ShapedType>();
  std::vector<Attribute> attribs;
  for (mlir::Value *a : ops) {
    ConstantOp def = cast<ConstantOp>(a->getDefiningOp());
    attribs.push_back(def.getValue());
  }
  return DenseElementsAttr::get(st, llvm::makeArrayRef(attribs));
}

#include "arc-opts.inc"

} // end anonymous namespace

void MakeVector::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  populateWithGenerated(context, &results);
}
