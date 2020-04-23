//===- Dialect definition for the Rust IR
//----------------------------------===//
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
// This file implements the IR Dialect for the Rust language.
//
//===----------------------------------------------------------------------===//

#ifndef RUST_DIALECT_H_
#define RUST_DIALECT_H_

#include "Rust/Types.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffects.h>

using namespace mlir;

namespace rust {

class RustPrinterStream;

#include "Rust/RustDialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// rust operations.
#define GET_OP_CLASSES
#include "Rust/Rust.h.inc"

LogicalResult writeModuleAsCrate(ModuleOp module, std::string top_dir,
                                 std::string rust_trailer,
                                 llvm::raw_ostream &o);

} // namespace rust

#endif // RUST_DIALECT_H_
