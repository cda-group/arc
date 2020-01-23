// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %1 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,f32> // expected-error {{'arc.make_tuple' op operand types do not match, found 'i1' but expected 'f32'}}
    return
  }
}
