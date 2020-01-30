//===- Arc IR Dialect registration in MLIR ------------------===//
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
//
// This file implements the dialect for the Arc IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "arc/arc-dialect.h"

using namespace mlir;
using namespace arc;

//===----------------------------------------------------------------------===//
// ArcDialect
//===----------------------------------------------------------------------===//
ArcDialect::ArcDialect(mlir::MLIRContext *ctx) : mlir::Dialect("arc", ctx) {
  addOperations<
#define GET_OP_LIST
#include "arc/ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//

LogicalResult IfOp::customVerify() {
  // We check that the result types of the blocks matche the result
  // type of the operator.
  auto Op = this->getOperation();
  auto ResultTy = Op->getResult(0).getType();
  bool FoundErrors = false;
  auto CheckResultType = [this, ResultTy, &FoundErrors](ArcBlockResultOp R) {
    if (R.getResult().getType() != ResultTy) {
      FoundErrors = true;
      emitOpError("result type does not match the type of the parent: found ")
          << R.getResult().getType() << " but expected " << ResultTy;
    }
  };

  for (unsigned RegionIdx = 0; RegionIdx < Op->getNumRegions(); RegionIdx++)
    Op->getRegion(RegionIdx).walk(CheckResultType);

  if (FoundErrors)
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "arc/ops.cpp.inc"
