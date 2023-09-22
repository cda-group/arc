// RUN: arc-mlir-rust-test %t %s
// RUN: arc-mlir-rust-test %t-canon %s -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {
  func.func @trigger_panic0() -> () {
    arc.panic()
    return
  }

  func.func @trigger_panic1() -> () {
    arc.panic("foo")
    return
  }
}
