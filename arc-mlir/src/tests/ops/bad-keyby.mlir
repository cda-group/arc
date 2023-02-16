// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// Check for an input which isn't a stream
module @toplevel {
  func.func @flip(%in : ui32) -> ui32 {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return %flipped : ui32
  }

  func.func @flip_keyby(%in : ui32)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op operand #0 must be a stream, but got 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (ui32) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> ui32 {
// expected-error@+2 {{'arc.keyby' op result #0 must be a stream, but got 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> ui32
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

  func.func @flip_keyby(%in : !arc.stream<ui32, si32>)
  	    -> !arc.stream<ui32, si32> {
// expected-error@+2 {{'arc.keyby' op keyby function type mismatch: input stream contains 'si32' but keyby function expects 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, si32>) -> !arc.stream<ui32, si32>
    return %out : !arc.stream<ui32, si32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, si32> {
// expected-error@+2 {{'arc.keyby' op input and output streams should have the same element type}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, si32>
    return %out : !arc.stream<ui32, si32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op keyby function expects environment but no 'key_fun_env_thunk' attribute set}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}
// -----

// Check for void keyby function
module @toplevel {
  func.func @flip(%in : ui32) {
    %one = arc.constant 0xFFFFFFFF: ui32
    %flipped = arc.and %in, %one : ui32
    return
  }

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op incorrect number of results for keyby function}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op keyby function does not expect an environment but 'key_fun_env_thunk' attribute set}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip, key_fun_env_thunk=@flip_env} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op keyby function environment type mismatch:}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip, key_fun_env_thunk=@flip_env} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}
// -----

// Check for a missing keyby function
module @toplevel {
  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.keyby' op 'flip' does not reference a valid function}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
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

  func.func @flip_keyby(%in : !arc.stream<ui32, ui32>)
      -> !arc.stream<si32, ui32> {
// expected-error@+2 {{'arc.keyby' op keyby function type mismatch: output stream key type is 'si32' but keyby function returns 'ui32'}}
// expected-note@+1 {{see current operation}}
    %out = "arc.keyby"(%in) { key_fun=@flip} : (!arc.stream<ui32, ui32>) -> !arc.stream<si32, ui32>
    return %out : !arc.stream<si32, ui32>
  }
}
