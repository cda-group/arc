// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 5 } : (tuple<i1,i1>) -> i1 // expected-error {{'arc.index_tuple' op index 5 is out-of-bounds for tuple with size 2}}

    return
  }
}
