// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func.func @ok0() -> !arc.adt<"i32"> {
    %out = arc.adt_constant "4711" : !arc.adt<"i32">
    return %out : !arc.adt<"i32">
  }

  func.func @ok1() -> !arc.adt<"(i32, bool)"> {
    %pair = arc.adt_constant "(17, false)" : !arc.adt<"(i32, bool)">
    return %pair : !arc.adt<"(i32, bool)">
  }
}
