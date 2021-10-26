//===-- Passes.td - Arc pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Arc
// directory. Boilerplate stolen from upstream.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_PASSES
#define ARC_PASSES

include "mlir/Pass/PassBase.td"

def LowerToRust : Pass<"arc-to-rust", "ModuleOp"> {
  let summary = "Convert arc/mlir to rust";
  let constructor = "arc::createLowerToRustPass()";

  let dependentDialects = [
        "arc::ArcDialect",
        "rust::RustDialect",
        "mlir::func::FuncDialect"
  ];
}

def RemoveSCF : Pass<"remove-scf", "mlir::func::FuncOp"> {
  let summary = "Convert structured flow-control to branches between basic blocks";
  let constructor = "arc::createRemoveSCFPass()";

  let dependentDialects = [
        "arc::ArcDialect",
        "mlir::func::FuncDialect"
  ];
}

def ToSCF : Pass<"to-scf", "ModuleOp"> {
  let summary = "Convert branches between basic blocks to structured flow-control";
  let constructor = "arc::createToSCFPass()";

  let dependentDialects = [
        "arc::ArcDialect",
        "mlir::func::FuncDialect"
  ];
}

def ToFSM : Pass<"to-fsm", "ModuleOp"> {
  let summary = "Convert functions with blocking operations into FSMs";
  let constructor = "arc::createToFSMPass()";

  let dependentDialects = [
        "arc::ArcDialect",
        "rust::RustDialect",
        "mlir::func::FuncDialect"
  ];
}
#endif // ARC_PASSES
