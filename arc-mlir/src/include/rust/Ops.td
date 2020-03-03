//===- Rust dialect operation definitions -------------------*- tablegen -*-===//
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
// Defines the operations of the Rust dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RUST_OPS
#define RUST_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Rust Dialect
//===----------------------------------------------------------------------===//

def Rust_Dialect : Dialect {
  let name = "rust";
  let cppNamespace = "rust";
}

//===----------------------------------------------------------------------===//
// Rust Operation Classes
//===----------------------------------------------------------------------===//

class Rust_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Rust_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//

def Rust_CrateOp : Rust_Op<"crate", [
  IsolatedFromAbove, SymbolTable, Symbol,
  SingleBlockImplicitTerminator<"CrateEndOp">
]> {
  let summary = "top-level rust crate";
  let description = [{}];
  let arguments = (ins);
  let regions = (region SizedRegion<1> : $body);
  let results = (outs);
  let extraClassDeclaration = [{
    LogicalResult customVerify();
    LogicalResult writeCrate(std::string, raw_ostream & os);
  }];
  let verifier = [{ return customVerify(); }];
}

def Rust_CrateEndOp : Rust_Op<"crate_end", [
  Terminator, HasParent<"CrateOp">
]> {
  let summary = "A pseudo op that marks the end of a rust.crate.";
  let description = [{
    This op terminates the only block inside the only region of a `rust.crate`.
  }];

  let parser = [{ return success(); }];
}

def Rust_RustFuncOp : Rust_Op<"func", [FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "A function";

  let description = [{
  }];

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the type of the function this Op defines.
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<RustFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();

    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustReturnOp : Rust_Op<"return", [HasParent<"RustFuncOp">,
    Terminator, SameOperandsAndResultType]> {
  let summary = "Terminator for Rust functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `rust.func`
    functions. The operands to the `rust.return` are the result values returned
    by an incovation of the `rust.func`.
  }];

  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

#endif // RUST_OPS
