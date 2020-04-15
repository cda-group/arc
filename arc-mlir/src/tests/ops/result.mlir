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
    %0 = "arc.make_appender"() : () -> !arc.appender<i32>
    // expected-error@+2 {{'arc.result' op result type does not match that of builder, found 'tensor<i64>' but expected 'tensor<i32>'}}
    // expected-note@+1 {{see current operation:}}
    %v = "arc.result"(%0) : (!arc.appender<i32>) -> tensor<i64>
    return
  }
}
