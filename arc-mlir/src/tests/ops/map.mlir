// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {

  func.func @_flip_env_thunk() -> !arc.struct<> {
    %env = arc.make_struct() : !arc.struct<>
    return %env : !arc.struct<>
  }

  func.func @flip(%in : ui32, %env : !arc.struct<>) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.map"(%in) { map_fun=@flip, map_fun_env_thunk=@_flip_env_thunk} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }

  func.func @flip_no_env(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map_no_env(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.map"(%in) { map_fun=@flip_no_env} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
