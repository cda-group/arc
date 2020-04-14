//===- Dialect definition for the Arc IR ----------------------------------===//
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
// This file implements the IR Dialect for the Arc language.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_DIALECT_H_
#define ARC_DIALECT_H_

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffects.h>

#include "Arc/Types.h"

using namespace mlir;

#include "Arc/ArcOpsEnums.h.inc"

namespace arc {

/// Include the auto-generated header file containing the declarations of the
/// arc operations.
#define GET_OP_CLASSES
#include "Arc/Arc.h.inc"
#include "Arc/ArcDialect.h.inc"

} // end namespace arc

#endif // ARC_DIALECT_H_
