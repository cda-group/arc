// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @ok0(%in : !arc.stream.source<si32>) -> !arc.stream.source<si32> {
    return %in : !arc.stream.source<si32>
  }

  func @ok1(%in : !arc.stream.sink<si32>) -> !arc.stream.sink<si32> {
    return %in : !arc.stream.sink<si32>
  }
}
