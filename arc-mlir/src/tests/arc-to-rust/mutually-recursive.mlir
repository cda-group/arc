// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @toplevel {

  func @is_even(%x : si32) -> i1 {
    %zero = arc.constant 0 : si32
    %cond = arc.cmpi "eq", %x, %zero : si32
    %res = "arc.if"(%cond) ({
      %true = arith.constant 1 : i1
      "arc.block.result"(%true) : (i1) -> ()
    }, {
      %one = arc.constant 1 : si32
      %y = arc.subi %x, %one : si32
      %res = call @is_odd(%y) : (si32) -> i1
      "arc.block.result"(%res) : (i1) -> ()
    }) : (i1) -> (i1)
    return %res : i1
  }

  func @is_odd(%x : si32) -> i1 {
    %zero = arc.constant 0 : si32
    %cond = arc.cmpi "eq", %x, %zero : si32
    %res = "arc.if"(%cond) ({
      %false = arith.constant 0 : i1
      "arc.block.result"(%false) : (i1) -> ()
    }, {
      %one = arc.constant 1 : si32
      %y = arc.subi %x, %one : si32
      %res = call @is_even(%y) : (si32) -> i1
      "arc.block.result"(%res) : (i1) -> ()
    }) : (i1) -> (i1)
    return %res : i1
  }

  func @ok0() -> () {
    %0 = constant @noop : () -> ()
    call @noop() : () -> ()
    return
  }

  func @noop() {
      return
  }

}
