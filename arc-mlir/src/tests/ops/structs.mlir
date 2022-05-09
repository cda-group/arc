// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func.func @ok0() -> !arc.struct<a : i32, b : f32> {
    %a = arith.constant 4 : i32
    %b = arith.constant 3.14 : f32
    %r = arc.make_struct(%a, %b : i32, f32) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }

  func.func @ok1() -> !arc.struct<a : i32> {
    %a = arith.constant 4 : i32
    %r = arc.make_struct(%a : i32) : !arc.struct<a : i32>
    return %r : !arc.struct<a : i32>
  }

  func.func @ok2() -> !arc.struct<a : i32, b : !arc.struct<a : i32> > {
    %a = arith.constant 4 : i32
    %b = arith.constant 3 : i32
    %s = arc.make_struct(%b : i32) : !arc.struct<a : i32>
    %r = arc.make_struct(%a, %s : i32, !arc.struct<a : i32>) : !arc.struct<a : i32, b : !arc.struct<a : i32>>
    return %r : !arc.struct<a : i32, b : !arc.struct<a : i32>>
  }

  func.func @ok6() -> si32 {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    %r_a = "arc.struct_access"(%r) { field = "a" } : (!arc.struct<a : si32, b : !arc.struct<a : si32>>) -> si32
    return %r_a : si32
  }

  func.func @ok7() -> si32 {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    %r_b = "arc.struct_access"(%r) { field = "b" } : (!arc.struct<a : si32, b : !arc.struct<a : si32>>) -> !arc.struct<a : si32>
    %r_b_a = "arc.struct_access"(%r_b) { field = "a" } : (!arc.struct<a : si32>) -> si32
    return %r_b_a : si32
  }

}
