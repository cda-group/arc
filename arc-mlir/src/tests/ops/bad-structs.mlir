// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = arith.constant 4 : i32
    %b = arith.constant 3.14 : f32
    // expected-error@+2 {{'arc.make_struct' op expected 2 fields, but found 0}}
    // expected-note@+1 {{see current operation}}
    %r = arc.make_struct() : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = arith.constant 4 : i32
    %b = arith.constant 3.14 : f32
    // expected-error@+2 {{'arc.make_struct' op expected 2 fields, but found 1}}
    // expected-note@+1 {{see current operation:}}
    %r = arc.make_struct(%a : i32) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = arith.constant 4 : i32
    %b = arith.constant 3.14 : f32
    %c = arith.constant 47.11 : f64
    // expected-error@+2 {{'arc.make_struct' op expected 2 fields, but found 3}}
    // expected-note@+1 {{see current operation:}}
    %r = arc.make_struct(%a, %b, %c: i32, f32, f64) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad() -> si32 {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    // expected-error@+2 {{'arc.struct_access' op field 'x' does not exist in}}
    // expected-note@+1 {{see current operation:}}
    %r_a = "arc.struct_access"(%r) { field = "x" } : (!arc.struct<a : si32, b : !arc.struct<a : si32>>) -> si32
    return %r_a : si32
  }
}

// -----

module @toplevel {
  func @bad() -> si32 {
    %a = arc.constant 4 : si32
    %b = arith.constant 3.14 : f32
    %r = arc.make_struct(%a, %b : si32, f32) : !arc.struct<a : si32, b : f32>
    // expected-error@+2 {{'arc.struct_access' op field 'b' does not have a matching type, expected 'si32' but found 'f32'}}
    // expected-note@+1 {{see current operation:}}
    %r_a = "arc.struct_access"(%r) { field = "b" } : (!arc.struct<a : si32, b : f32>) -> si32
    return %r_a : si32
  }
}

// -----
