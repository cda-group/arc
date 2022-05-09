// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func.func @bad0(%v : si64, %s : !arc.stream.sink<si32>) -> ()
  attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.send' op Can't send value of type 'si64' on a '!arc.stream.sink<si32>' stream}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si64, !arc.stream.sink<si32>) -> ()
    return
  }
}

// -----

module @toplevel {
  func.func @bad1(%v : si64, %s : !arc.stream.source<si64>) -> ()
  attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.send' op operand #1 must be a sink stream, but got '!arc.stream.source<si64>'}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si64, !arc.stream.source<si64>) -> ()
    return
  }
}

// -----

module @toplevel {
  func.func @bad2(%v : si32, %s : !arc.stream.sink<si32>) -> () {
// expected-error@+2 {{'arc.send' op can only be used inside a task}}
// expected-note@+1 {{see current operation: "arc.send"}}
    "arc.send"(%v, %s) : (si32, !arc.stream.sink<si32>) -> ()
    return
  }
}

// -----

module @toplevel {
  func.func @bad3(%s : !arc.stream.sink<si32>) -> si32
    attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.receive' op operand #0 must be a source stream, but got '!arc.stream.sink<si32>'}}
// expected-note@+1 {{see current operation:}}
    %v = "arc.receive"(%s) : (!arc.stream.sink<si32>) -> si32
    return %v : si32
  }
}

// -----

module @toplevel {
  func.func @bad3(%s : !arc.stream.source<si32>) -> si64
    attributes { "arc.is_task" } {
// expected-error@+2 {{'arc.receive' op Can't receive a value of type 'si64' from a '!arc.stream.source<si32>' stream}}
// expected-note@+1 {{see current operation:}}
    %v = "arc.receive"(%s) : (!arc.stream.source<si32>) -> si64
    return %v : si64
  }
}

