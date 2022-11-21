// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// Check for an input which isn't a stream
module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : ui32)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op operand #0 must be a source stream, but got 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (ui32) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
// -----

// Check for an output which isn't a stream
module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> ui32 {
// expected-error@+2 {{'arc.map' op result #0 must be a source stream, but got 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> ui32
    return %out : ui32
  }
}
// -----

// Check for input type mismatch
module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<si32>)
  	    -> !arc.stream.source<si32> {
// expected-error@+2 {{'arc.map' op map function type mismatch: input stream contains 'si32' but map function expects 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<si32>) -> !arc.stream.source<si32>
    return %out : !arc.stream.source<si32>
  }
}
// -----

// Check for output type mismatch
module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<si32> {
// expected-error@+2 {{'arc.map' op map function type mismatch: output stream contains 'si32' but map function returns 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> !arc.stream.source<si32>
    return %out : !arc.stream.source<si32>
  }
}
// -----

// Check for missing environment
module @toplevel {
  func.func @flip(%in : ui32, %extra : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op map function expects environment but no 'map_fun_env_thunk' attribute set}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
// -----

// Check for void map function
module @toplevel {
  func.func @flip(%in : ui32) {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op incorrect number of results for map function}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
// -----

// Check for redundant environment
module @toplevel {
  func.func @flip_env() -> ui32 {
    %env = arc.constant 0xFFFFFFFF: ui32
    return %env : ui32
  }

  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op map function does not expect an environment but 'map_fun_env_thunk' attribute set}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip, map_fun_env_thunk=@flip_env} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
// -----
// Check for type mismatched environment
module @toplevel {
  func.func @flip_env() -> si32 {
    %env = arc.constant 0 : si32
    return %env : si32
  }

  func.func @flip(%in : ui32, %extra : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op map function environment type mismatch:}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip, map_fun_env_thunk=@flip_env} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
// -----

// Check for a missing map function
module @toplevel {
  func.func @flip_map(%in : !arc.stream.source<ui32>)
  	    -> !arc.stream.source<ui32> {
// expected-error@+2 {{'arc.map' op 'flip' does not reference a valid function}}
// expected-note@+1 {{see current operation}}
    %out = "arc.map"(%in) { map_fun=@flip} : (!arc.stream.source<ui32>) -> !arc.stream.source<ui32>
    return %out : !arc.stream.source<ui32>
  }
}
