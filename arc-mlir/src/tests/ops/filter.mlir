// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @is_even(%in : ui32) -> i1 {
    %zero = arc.constant 0: ui32
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %zero : ui32
    return %bool : i1
  }

  func.func @odd_filter(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }

  func.func @even_filter(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.filter"(%in) { predicate=@is_even} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }

  func.func @chained_filter(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %t = "arc.filter"(%in) { predicate=@is_even} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    %r = "arc.filter"(%t) { predicate=@is_odd} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %r : !arc.stream.source<ui32>
  }

  func.func @is_same_with_env(%in : ui32, %env : ui32) -> i1 {
    %bool = arc.cmpi "eq", %in, %env : ui32
    return %bool : i1
  }

  func.func @one_thunk() -> ui32 {
    %t = arc.constant 1: ui32
    return %t : ui32
  }


  func.func @filter_with_env(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
    %out = "arc.filter"(%in) { predicate=@is_same_with_env, predicate_env_thunk=@one_thunk } : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }


}
