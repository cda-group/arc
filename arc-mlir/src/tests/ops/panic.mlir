// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

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
