// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @empty_struct(%in : !arc.struct<>) -> !arc.struct<> {
    return %in : !arc.struct<>
  }

  func @make_empty_struct() -> !arc.struct<> {
    %r = arc.make_struct() : !arc.struct<>
    return %r : !arc.struct<>
  }

}
