// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<i1,i1>) -> f64 // expected-error {{'arc.index_tuple' op element type at index 1 does not match result, found 'i1' but expected 'f64'}}

    return
  }
}

