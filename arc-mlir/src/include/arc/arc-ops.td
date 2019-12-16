//===- Arc dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_OPS
#define ARC_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Arc_Dialect : Dialect {
  let name = "arc";
  let cppNamespace = "arc";
}

class Arc_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Arc_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//
def MakeVector
    : Arc_Op<"make_vector", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "create a vector from a sequence of values";
  let description =
      [{A pure operation which turns its arguments into a vector}];

  let arguments = (ins Variadic<AnyType> : $values);

  let results = (outs 1DTensorOf<[AnyType]>);

  let verifier = [{
    unsigned NumOperands = this->getOperation()->getNumOperands();
    auto ElemTy = this->getOperation()->getOperand(0)->getType();
    auto TensorTy =
        this->getOperation()->getResult(0)->getType().dyn_cast<TensorType>();
    if (!TensorTy.hasStaticShape())
      return emitOpError("result must have static shape, expected ")
             << RankedTensorType::get({NumOperands}, ElemTy);
    if (NumOperands != TensorTy.getNumElements())
      return emitOpError("result does not match the number of operands: found ")
             << NumOperands << " but expected " << TensorTy.getNumElements()
             << " operands";
    return mlir::success();
  }];
}

def ArcKeep : Arc_Op<"keep", []> {
  let summary = "Operation consuming anything and having side effects.";
  let description = [{An operation used for debugging and testing. Can be used
                      to prevent optimization passes from removing dummy
                      results.
  }];
  let arguments = (ins Variadic<AnyType> : $values);
  let results = (outs);
}

#endif // ARC_OPS
