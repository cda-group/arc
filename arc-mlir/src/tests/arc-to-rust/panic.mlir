// RUN: arc-mlir-rust-test %t %s
// RUN: arc-mlir-rust-test %t-canon %s -canonicalize

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
