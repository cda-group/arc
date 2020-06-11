// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %a = constant false
    %b = constant true

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 0 } : (tuple<i1,i1>) -> i1

    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant false
    %b = constant true
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+2 {{'arc.index_tuple' op element type at index 1 does not match result: expected 'f64' but found 'i1'}}
    // expected-note@+1 {{see current operation:}}
    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<i1,i1>) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant false
    %b = constant true
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+1 {{'arc.index_tuple' op attribute 'index' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
    %elem = "arc.index_tuple"(%tuple) { index = -5 } : (tuple<i1,i1>) -> i1
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant false
    %b = constant true
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+1 {{'arc.index_tuple' op requires attribute 'index'}}
    %elem = "arc.index_tuple"(%tuple) : (tuple<i1,i1>) -> i1
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant false
    %b = constant true
    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>

    // expected-error@+2 {{'arc.index_tuple' op index 5 is out-of-bounds for tuple with size 2}}
    // expected-note@+1 {{see current operation:}}
    %elem = "arc.index_tuple"(%tuple) { index = 5 } : (tuple<i1,i1>) -> i1
    return
  }
}
