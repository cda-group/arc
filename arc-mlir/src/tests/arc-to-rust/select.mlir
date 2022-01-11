// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustifs  {
  func @test_0(%arg0: i1, %arg1: ui32, %arg2: ui32) -> ui32 {
    %0 = select %arg0, %arg1, %arg2 : ui32
    return %0 : ui32
  }
}
