// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @toplevel {
  func private @new(%str : !arc.adt<"Text">) -> !arc.adt<"Str">

  func private @print(%str : !arc.adt<"Str">) -> ()

  func private @append(%s0 : !arc.adt<"Str">, %s1 : !arc.adt<"Str">) -> ()

  func @ok0() -> () {
    %0 = arc.adt_constant "\"Hello World\"" : !arc.adt<"Text">
    %1 = call @new(%0) : (!arc.adt<"Text">) -> !arc.adt<"Str">
    call @print(%1) : (!arc.adt<"Str">) -> ()
    return
  }

  func @ok1() -> () {
    %0 = arc.adt_constant "\"Hello \"" : !arc.adt<"Text">
    %1 = arc.adt_constant "\"World\"" : !arc.adt<"Text">
    %2 = call @new(%0) : (!arc.adt<"Text">) -> !arc.adt<"Str">
    %3 = call @new(%1) : (!arc.adt<"Text">) -> !arc.adt<"Str">
    call @append(%2, %3) : (!arc.adt<"Str">, !arc.adt<"Str">) -> ()
    call @print(%2) : (!arc.adt<"Str">) -> ()
    return
  }
}
