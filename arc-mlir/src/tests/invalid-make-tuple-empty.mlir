// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %1 = "arc.make_tuple"() : () -> tuple<> // expected-error {{'arc.make_tuple' op tuple must contain at least one element}}
    return
  }
}
