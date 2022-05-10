// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
// expected-error@+1 {{expected ':'}}
  func.func @fail1(%in : !arc.struct<foo>) -> () {
    return
  }
}

// -----

module @toplevel {
// expected-error@+1 {{expected non-function type}}
  func.func @fail2(%in : !arc.struct<foo : >) -> () {
    return
  }
}

// -----

module @toplevel {
// expected-note@+1 {{prior use here}}
  func.func @fail3(%in : !arc.struct<foo : i32, bar : f32,
                              inner_struct : !arc.struct<nested : i32>>) -> () {
// expected-error@+1 {{use of value '%in' expects different type than prior uses: '!arc.struct<foo : i32, bar : f32>' vs '!arc.struct<foo : i32, bar : f32, inner_struct : !arc.struct<nested : i32>>'}}
    return %in : !arc.struct<foo : i32, bar : f32>
  }
}

// -----

module @toplevel {
  func.func @fail4(%in : !arc.struct<foo : i32, bar : f32>) -> !arc.struct<foo : i32, baz : f32> {
  // expected-error@+2 {{type of return operand 0 ('!arc.struct<foo : i32, bar : f32>') doesn't match function result type ('!arc.struct<foo : i32, baz : f32>')}}
  // expected-note@+1 {{see current operation: "func.return"(%arg0) : (!arc.struct<foo : i32, bar : f32>) -> ()}}
    return %in : !arc.struct<foo : i32, bar : f32>
  }
}
