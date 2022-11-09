// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
