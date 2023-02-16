// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// Check for mismatch between stream and predicate
module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, si32>)
  	    -> !arc.stream<ui32, si32> {
// expected-error@+2 {{'arc.filter' op predicate type mismatch: expected operand type 'ui32', but received'si32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, si32>) -> !arc.stream<ui32, si32>
    return %out : !arc.stream<ui32, si32>
  }

}

// -----
// Check for mismatch between the type of the input and output streams
module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, si32> {
// expected-error@+2 {{'arc.filter' op input and output streams should have the same types}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, si32>
    return %out : !arc.stream<ui32, si32>
  }

}

// -----
// Check for a predicate which returns a non-boolean
module @toplevel {
  func.func @is_odd(%in : ui32) -> ui32 {
    return %in : ui32
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.filter' op predicate does not return a boolean}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }

}

// -----
// Check for a predicate which doesn't return a value
module @toplevel {
  func.func @is_odd(%in : ui32)  {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return
  }
  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.filter' op incorrect number of results for predicate}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}

// -----
// Check for a predicate with too few arguments
module @toplevel {
  func.func @is_odd() -> i1 {
    %one = arith.constant 1: i1
    return %one : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.filter' op incorrect number of operands for predicate}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}

// -----
// Check for a predicate with too many arguments
module @toplevel {
  func.func @is_odd(%elem : ui32, %env : ui32, %extra : ui32) -> i1 {
    %one = arith.constant 1: i1
    return %one : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.filter' op incorrect number of operands for predicate}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}

// -----
// Check for an input which isn't a stream
module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

// expected-note@+1 {{prior use here}}
  func.func @odd_filter(%in : ui32)
  	    -> !arc.stream<ui32, si32> {
// expected-error@+1 {{use of value '%in' expects different type than prior uses: '!arc.stream<ui32, ui32>' vs 'ui32'}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, si32>
    return %out : !arc.stream<ui32, si32>
  }

}
// -----
// Check for an output which isn't a stream
module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> si32 {
// expected-note@+1 {{prior use here}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, si32>
// expected-error@+1 {{use of value '%out' expects different type than prior uses: 'si32' vs '!arc.stream<ui32, si32>}}
    return %out : si32
  }

}

// -----
// Check for a missing environment in the filter call
module @toplevel {
  func.func @is_odd(%in : ui32, %env : si32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-note@+2 {{see current}}
// expected-error@+1 {{'arc.filter' op predicate expects environment but no 'predicate_env_thunk' attribute set}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }

}

// -----
// Check for a not needed environment in the filter call
module @toplevel {
  func.func @is_odd(%in : ui32) -> i1 {
    %one = arc.constant 1: ui32
    %bit = arc.and %in, %one : ui32
    %bool = arc.cmpi "eq", %bit, %one : ui32
    return %bool : i1
  }

  func.func @dummy_env() -> () {
    return
  }

  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
    %one = arc.constant 1: ui32
// expected-note@+2 {{see current}}
// expected-error@+1 {{'arc.filter' op map function does not expect an environment but 'predicate_env_thunk' attribute set}}
    %out = "arc.filter"(%in) { predicate=@is_odd, predicate_env_thunk=@dummy_env} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }

}

// -----
// Check for an environment mismatch
module @toplevel {

  func.func @is_same_with_env(%in : ui32, %env : ui32) -> i1 {
    %bool = arc.cmpi "eq", %in, %env : ui32
    return %bool : i1
  }

  func.func @thunk() -> si32 {
    %t = arc.constant 1: si32
    return %t : si32
  }

  func.func @filter_with_env(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-note@+2 {{see current}}
// expected-error@+1 {{'arc.filter' op predicate environment type mismatch:}}
    %out = "arc.filter"(%in) { predicate=@is_same_with_env, predicate_env_thunk=@thunk} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}

// -----
// Check for a missing predicate
module @toplevel {
  func.func @odd_filter(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.filter' op 'is_odd' does not reference a valid function}}
// expected-note@+1 {{see current operation}}
    %out = "arc.filter"(%in) { predicate=@is_odd} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }

}
