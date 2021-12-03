// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize


module @toplevel {
  func @and_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.andi %arg0, %arg1 : i1
    return %0 : i1
  }
  func @or_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.ori %arg0, %arg1 : i1
    return %0 : i1
  }
  func @xor_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.xori %arg0, %arg1 : i1
    return %0 : i1
  }
}
