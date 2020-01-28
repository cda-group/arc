// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %1 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1> // expected-error {{'arc.make_tuple' op result does not match the number of operands: found 2 but expected 1 operands}}
    return
  }
}
