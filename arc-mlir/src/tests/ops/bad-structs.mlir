// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = constant 4 : i32
    %b = constant 3.14 : f32
    // expected-error@+1 {{expected ':'}}
    %r = arc.make_struct() : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = constant 4 : i32
    %b = constant 3.14 : f32
    // expected-error@+2 {{'arc.make_struct' op expected 2 fields, but found 1}}
    // expected-note@+1 {{see current operation: %0 = "arc.make_struct"(%c4_i32) : (i32) -> !arc.struct<a : i32, b : f32>}}
    %r = arc.make_struct(%a : i32) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----

module @toplevel {
  func @bad0() -> !arc.struct<a : i32, b : f32> {
    %a = constant 4 : i32
    %b = constant 3.14 : f32
    %c = constant 47.11 : f64
    // expected-error@+2 {{'arc.make_struct' op expected 2 fields, but found 3}}
    // expected-note@+1 {{see current operation: %0 = "arc.make_struct"(%c4_i32, %cst, %cst_0) : (i32, f32, f64) -> !arc.struct<a : i32, b : f32>}}
    %r = arc.make_struct(%a, %b, %c: i32, f32, f64) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }
}

// -----
