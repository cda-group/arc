// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %r = "arc.result_appender"(%0) : (!arc.appender<i32>) -> tensor<?xi32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    // expected-error@+1 {{'arc.result_appender' op element type of tensor does not match merge type of appender, found 'i64' but expected 'i32'}}
    %v = "arc.result_appender"(%0) : (!arc.appender<i32>) -> tensor<?xi64>
    return
  }
}
