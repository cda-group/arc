// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main(%arg0: !arc.appender<i32>) {
    return
  }
}

// -----

module @toplevel {
  func @main(%arg0: !arc.appender<tuple<i32, i32, f64>>) {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+1 {{appender merge type must be a value type: found '!arc.appender<i32>'}}
  func @main(%arg0: !arc.appender<!arc.appender<i32>>) {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+1 {{appender merge type must be a value type: found 'tuple<i32, !arc.appender<i32>>'}}
  func @main(%arg0: !arc.appender<tuple<i32, !arc.appender<i32>>>) {
    return
  }
}
