// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @bad0(%v : si64, %s : !arc.stream.sink<si32>) -> ()
  attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.send' op Can't send value of type 'si64' on a '!arc.stream.sink<si32>' stream}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si64, !arc.stream.sink<si32>) -> ()
    return
  }
}

// -----

module @toplevel {
  func @bad1(%v : si64, %s : !arc.stream.source<si64>) -> ()
  attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.send' op operand #1 must be a sink stream, but got '!arc.stream.source<si64>'}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si64, !arc.stream.source<si64>) -> ()
    return
  }
}

// -----

module @toplevel {
  func @bad2(%v : si32, %s : !arc.stream.sink<si32>) -> () {
// expected-error@+2 {{'arc.send' op can only be used inside a task}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si32, !arc.stream.sink<si32>) -> ()
    return
  }
}
