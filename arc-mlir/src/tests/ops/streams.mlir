// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func.func @ok0(%in : !arc.stream.source<ui32, si32>) -> !arc.stream.source<ui32, si32>
    attributes { "arc.is_task" } {
    return %in : !arc.stream.source<ui32, si32>
  }

  func.func @ok1(%in : !arc.stream.sink<ui32, si32>) -> !arc.stream.sink<ui32, si32>
    attributes { "arc.is_task" } {
    return %in : !arc.stream.sink<ui32, si32>
  }

  func.func @ok2(%v : si32, %s : !arc.stream.sink<ui32, si32>) -> ()
    attributes { "arc.is_task" } {
    "arc.send"(%v, %s) : (si32, !arc.stream.sink<ui32, si32>) -> ()
    return
  }

  func.func @ok3(%s : !arc.stream.source<ui32, si32>) -> si32
    attributes { "arc.is_task" } {
    %v = "arc.receive"(%s) : (!arc.stream.source<ui32, si32>) -> si32
    return %v : si32
  }
}
