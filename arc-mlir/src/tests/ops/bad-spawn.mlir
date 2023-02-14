// RUN: arc-mlir %s -split-input-file -verify-diagnostics
module @arctorustspawn {

  func.func @id(%in : !arc.stream.source<ui32, si32>,
                %out : !arc.stream.sink<ui32, si32>) -> ()
    attributes {
      arc.is_task,
      rust.async,
      rust.annotation = "#[rewrite(nonpersistent)]"
    } {
      scf.while () : () -> () {
        %condition = arith.constant 1 : i1
        scf.condition(%condition)
      } do {
        ^bb0():
        %x = "arc.receive"(%in) : (!arc.stream.source<ui32, si32>) -> si32
        "arc.send"(%x, %out) : (si32, !arc.stream.sink<ui32, si32>) -> ()
        scf.yield
      }
      return
    }

    func.func @main(%in : !arc.stream.source<ui32, ui32>,
                    %out : !arc.stream.sink<ui32, ui32>) {
    // expected-error@+2 {{'arc.spawn' op operand type mismatch}}
    // expected-note@+1 {{see current operation:}}
       arc.spawn @id(%in, %out) : (!arc.stream.source<ui32, ui32>,
                                   !arc.stream.sink<ui32, ui32>) -> ()
       return
    }
}

// -----

module @arctorustspawn {

  func.func @id(%in : !arc.stream.source<ui32, si32>,
                %out : !arc.stream.sink<ui32, si32>) -> (si32)
    attributes {
      arc.is_task,
      rust.async,
      rust.annotation = "#[rewrite(nonpersistent)]"
    } {
      scf.while () : () -> () {
        %condition = arith.constant 1 : i1
        scf.condition(%condition)
      } do {
        ^bb0():
        %x = "arc.receive"(%in) : (!arc.stream.source<ui32, si32>) -> si32
        "arc.send"(%x, %out) : (si32, !arc.stream.sink<ui32, si32>) -> ()
        scf.yield
      }
      %r = arc.constant 1 : si32
      return %r : si32
    }

    func.func @main(%in : !arc.stream.source<ui32, si32>,
                    %out : !arc.stream.sink<ui32, si32>) {
    // expected-error@+2 {{op 'callee' should not have a result}}
    // expected-note@+1 {{see current operation:}}
       arc.spawn @id(%in, %out) : (!arc.stream.source<ui32, si32>,
                                   !arc.stream.sink<ui32, si32>) -> ()
       return
    }
}

// -----

module @arctorustspawn {

  func.func @id(%in : !arc.stream.source<ui32, si32>,
                %out : !arc.stream.sink<ui32, si32>) -> ()
    attributes {
      rust.async,
      rust.annotation = "#[rewrite(nonpersistent)]"
    } {
      return
    }

    func.func @main(%in : !arc.stream.source<ui32, si32>,
                    %out : !arc.stream.sink<ui32, si32>) {
// expected-error@+2 {{'arc.spawn' op 'callee' must be a task}}
// expected-note@+1 {{see current operation:}}
       arc.spawn @id(%in, %out) : (!arc.stream.source<ui32, si32>,
                                   !arc.stream.sink<ui32, si32>) -> ()
       return
    }
}
