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

  func.func @flip_map(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
    %out = "arc.keyby"(%in) { key_fun=@flip, key_fun_env_thunk=@_flip_env_thunk} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }

  func.func @flip_no_env(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map_no_env(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
    %out = "arc.keyby"(%in) { key_fun=@flip_no_env} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}
