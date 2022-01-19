// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @id(%a : !arc.stream.source<si32>,
           %b : !arc.stream.source<si32>,
           %c : !arc.stream.sink<si32>,
           %d : !arc.stream.sink<si32>) -> ()
    attributes { "arc.is_task" } {
    %x = "arc.receive"(%a) : (!arc.stream.source<si32>) -> si32
    %y = "arc.receive"(%b) : (!arc.stream.source<si32>) -> si32
    "arc.send"(%x, %c) : (si32, !arc.stream.sink<si32>) -> ()
    "arc.send"(%y, %d) : (si32, !arc.stream.sink<si32>) -> ()
    return
  }
}
