// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {

  func @ok2(%in : !arc.adt<"i16">) -> !arc.adt<"i32"> {
  // expected-error@+2 {{type of return operand 0 ('!arc.adt<"i16">') doesn't match function result type ('!arc.adt<"i32">')}}
  // expected-note@+1 {{see current operation}}
    return %in : !arc.adt<"i16">
  }
}

// -----
