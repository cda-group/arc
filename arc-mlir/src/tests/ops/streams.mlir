// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @ok0(%in : !arc.stream.source<si32>) -> !arc.stream.source<si32>
    attributes { "arc.is_task" } {
    return %in : !arc.stream.source<si32>
  }

  func @ok1(%in : !arc.stream.sink<si32>) -> !arc.stream.sink<si32>
    attributes { "arc.is_task" } {
    return %in : !arc.stream.sink<si32>
  }

  func @ok2(%v : si32, %s : !arc.stream.sink<si32>) -> ()
    attributes { "arc.is_task" } {
    "arc.send"(%v, %s) : (si32, !arc.stream.sink<si32>) -> ()
    return
  }

  func @ok3(%s : !arc.stream.source<si32>) -> si32
    attributes { "arc.is_task" } {
    %v = "arc.receive"(%s) : (!arc.stream.source<si32>) -> si32
    return %v : si32
  }
}
