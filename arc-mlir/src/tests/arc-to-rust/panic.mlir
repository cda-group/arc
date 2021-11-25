// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @toplevel {
  func @trigger_panic0() -> () {
    arc.panic()
    return
  }

  func @trigger_panic1() -> () {
    arc.panic("foo")
    return
  }

}
