// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @bad() -> !arc.enum<a : i32, b : f32> {
    %a = constant 4 : i32
    // expected-error@+2 {{'arc.make_enum' op : variant 'c' does not exist in '!arc.enum<a : i32, b : f32>'}}
    // expected-note@+1 {{see current operation: %0 = "arc.make_enum"(%c4_i32) {variant = "c"} : (i32) -> !arc.enum<a : i32, b : f32>}}
    %r = arc.make_enum (%a : i32) as "c" : !arc.enum<a : i32, b : f32>
    return %r : !arc.enum<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad() -> !arc.enum<a : i32, b : f32> {
    %a = constant 3.14 : f32
    // expected-error@+2 {{'arc.make_enum' op : variant 'a' does not have a matching type, expected 'f32' but found 'i32'}}
    // expected-note@+1 {{see current operation: %0 = "arc.make_enum"(%cst) {variant = "a"} : (f32) -> !arc.enum<a : i32, b : f32>}}
    %r = arc.make_enum (%a : f32) as "a" : !arc.enum<a : i32, b : f32>
    return %r : !arc.enum<a : i32, b : f32>
  }
}
