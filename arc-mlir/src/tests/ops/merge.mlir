// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %v = constant 5 : i32
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %1 = "arc.merge"(%0, %v) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %r = "arc.result"(%1) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %v = constant 5 : i64
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>

    // expected-error@+1 {{'arc.merge' op operand type does not match merge type, found 'i64' but expected 'i32'}}
    %1 = "arc.merge"(%0, %v) : (!arc.appender<i32>, i64) -> !arc.appender<i32>

    %r = "arc.result"(%1) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %v = constant 5 : i32

    // expected-error@+1 {{'arc.merge' op result type does not match builder type, found: '!arc.appender<i64>' but expected '!arc.appender<i32>'}}
    %1 = "arc.merge"(%0, %v) : (!arc.appender<i32>, i32) -> !arc.appender<i64>

    %c = "arc.result"(%1) : (!arc.appender<i64>) -> tensor<i64>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %v1 = constant 5 : i32
    %v2 = constant 5 : i32

    // expected-error@+1 {{'arc.make_appender' op failed to verify that result must have exactly one use}}
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>

    %1 = "arc.merge"(%0, %v1) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %2 = "arc.merge"(%0, %v2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %r1 = "arc.result"(%1) : (!arc.appender<i32>) -> tensor<i32>
    %r2 = "arc.result"(%2) : (!arc.appender<i32>) -> tensor<i32>
    return
  }
}
