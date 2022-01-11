// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustfuncarguments {
  func @zero_args() -> ui16 {
    %t = arc.constant 4711 : ui16
    return %t : ui16
  }

  func @one_arg(%a : ui16) -> ui16 {
    return %a : ui16
  }

  func @two_args(%a : ui16, %b : ui16) -> ui16 {
    return %b : ui16
  }
}
