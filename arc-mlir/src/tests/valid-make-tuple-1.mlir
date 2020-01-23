// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %c = constant 0 : i1

    %1 = "arc.make_tuple"(%a, %b, %c) : (i1, i1, i1) -> tuple<i1,i1,i1>
    return
  }
}
