// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %false_4 = constant 0 : i1
    %true_5 = constant 1 : i1
    %false_6 = constant 0 : i1

    %1 = "arc.make_vector"(%false_4, %false_4, %true_5, %false_6) : (i1, i1, i1, i1) -> tensor<5xi1> // expected-error {{'arc.make_vector' op result does not match the number of operands: found 4 but expected 5 operands}}
    return
  }
}