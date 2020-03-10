//===- Pattern Match Optimizations for ARC -----------------*- tablegen -*-===//
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

#ifndef ARC_OPTS
#define ARC_OPTS

#ifndef STANDARD_OPS
include "../../mlir/include/mlir/Dialect/StandardOps/IR/Ops.td"
#endif // STANDARD_OPS

#ifndef ARC_OPS
include "../../include/Arc/ArcDialect.td"
#endif // ARC_OPS

def AllValuesAreConstants : Constraint<CPred<"AllValuesAreConstant($0)">>;
def ConstantValuesToDenseAttributes :
    NativeCodeCall<"ConstantValuesToDenseAttributes($0, $1)">;
def ConstantMakeVectorPattern : Pat<(MakeVectorOp:$result $values),
          (ConstantOp (ConstantValuesToDenseAttributes $result, $values)),
          [(AllValuesAreConstants $values)]>;

#endif // ARC_OPTS
