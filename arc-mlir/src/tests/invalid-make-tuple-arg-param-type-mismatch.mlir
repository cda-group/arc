// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %a = constant -3.40282347E+38 : f32 // expected-note {{prior use here}}
    %b = constant 0 : i1
    %0 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1> // expected-error {{use of value '%a' expects different type than prior uses: 'i1' vs 'f32'}}
    return
  }
}
