// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func.func @main(%arg0: !arc.appender<i32>) {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+1 {{appender merge type must be a value type: found '!arc.appender<i32>'}}
  func.func @main(%arg0: !arc.appender<!arc.appender<i32>>) {
    return
  }
}
