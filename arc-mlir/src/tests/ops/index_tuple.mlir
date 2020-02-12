// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 0 } : (tuple<i1,i1>) -> i1

    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+1 {{'arc.index_tuple' op element type at index 1 does not match result, found 'i1' but expected 'f64'}}
    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<i1,i1>) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+1 {{'arc.index_tuple' op attribute 'index' failed to satisfy constraint: non-negative 64-bit integer attribute}}
    %elem = "arc.index_tuple"(%tuple) { index = -5 } : (tuple<i1,i1>) -> i1
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error {{'arc.index_tuple' op requires attribute 'index'}}
    %elem = "arc.index_tuple"(%tuple) : (tuple<i1,i1>) -> i1
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error {{'arc.index_tuple' op index 5 is out-of-bounds for tuple with size 2}}
    %elem = "arc.index_tuple"(%tuple) { index = 5 } : (tuple<i1,i1>) -> i1
    return
  }
}
