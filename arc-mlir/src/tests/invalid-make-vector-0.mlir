// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @main() {
    %cst_0 = constant -3.40282347E+38 : f32 // expected-note {{prior use here}}
    %false_4 = constant 0 : i1
    %true_5 = constant 1 : i1
    %false_6 = constant 0 : i1
    %0 = "arc.make_vector"(%cst_0, %false_4, %true_5, %false_6) : (i1, i1, i1, i1) -> tensor<4xi1> // expected-error {{use of value '%cst_0' expects different type than prior uses: 'i1' vs 'f32'}}
    return
  }
}
