// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize


module @toplevel {
  func.func @and_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.andi %arg0, %arg1 : i1
    return %0 : i1
  }
  func.func @or_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.ori %arg0, %arg1 : i1
    return %0 : i1
  }
  func.func @xor_i1(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.xori %arg0, %arg1 : i1
    return %0 : i1
  }
  func.func @eq_i1(%a : i1, %b : i1) -> i1 {
    %r = arith.cmpi "eq", %a, %b : i1
    return %r : i1
  }
  func.func @ne_i1(%a : i1, %b : i1) -> i1 {
    %r = arith.cmpi "ne", %a, %b : i1
    return %r : i1
  }
  func.func @not_i1(%arg0: i1) -> i1 {
    %0 = arith.constant 1 : i1
    %1 = arith.xori %arg0, %0 : i1
    return %1 : i1
  }
  func.func @not_select_i1(%arg0: i1) -> i1 {
    %0 = arith.constant 0 : i1
    %1 = arith.constant 1 : i1
    %2 = arith.select %arg0, %0, %1 : i1
    return %2 : i1
  }
}
