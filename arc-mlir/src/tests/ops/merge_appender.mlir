// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %v = constant 5 : i32
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %1 = "arc.merge_appender"(%0, %v) : (!arc.appender<i32>, i32) -> i32
    %r = "arc.result_appender"(%1) : (!arc.appender<i32>) -> tensor<?xi32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %v = constant 5 : i64
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>

    // expected-error@+1 {{'arc.merge_appender' op value type does not match merge type, found 'i64' but expected 'i32'}}
    %1 = "arc.merge_appender"(%0, %v) : (!arc.appender<i32>, i64) -> !arc.appender<i32>

    %r = "arc.result_appender"(%1) : (!arc.appender<i32>) -> tensor<?xi32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %v = constant 5 : i32

    // expected-error@+1 {{'arc.merge_appender' op inferred type incompatible with return type of operation}}
    %1 = "arc.merge_appender"(%0, %v) : (!arc.appender<i32>, i32) -> !arc.appender<i64>

    %c = "arc.result_appender"(%1) : (!arc.appender<i64>) -> tensor<?xi64>
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

    %1 = "arc.merge_appender"(%0, %v1) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %2 = "arc.merge_appender"(%0, %v2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %r1 = "arc.result_appender"(%1) : (!arc.appender<i32>) -> tensor<?xi32>
    %r2 = "arc.result_appender"(%2) : (!arc.appender<i32>) -> tensor<?xi32>
    return
  }
}
