// Check that round-tripping works for the ADT type
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {

  func @ok0(%in : !arc.adt<"i32">) -> () {
    return
  }

  func @ok2(%in : !arc.adt<"i32">) -> !arc.adt<"i32"> {
    return %in : !arc.adt<"i32">
  }
}
