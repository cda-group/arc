// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %r = "arc.result"(%0) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %v = constant 0 : i32

    // expected-error@+2 {{'arc.make_appender' op requires zero operands}}
    // expected-note@+1 {{see current operation:}}
    %0 = "arc.make_appender"(%v) : (i32) -> !arc.appender<i32>

    %r = "arc.result"(%0) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    // SHOULD FAIL
    %0 = "arc.make_appender"() {size = -1} : () -> !arc.appender<i32>
    %b = "arc.result"(%0) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    // expected-error@+2 {{'arc.make_appender' op result #0 must be any appender, but got 'i32'}}
    // expected-note@+1 {{see current operation: %0 = "arc.make_appender"}}
    %0 = "arc.make_appender"() : () -> i32
    %b = constant 2 : i32
    %c = addi %0, %b : i32
    return
  }
}
