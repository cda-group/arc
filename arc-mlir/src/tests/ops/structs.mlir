// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @ok0() -> !arc.struct<a : i32, b : f32> {
    %a = constant 4 : i32
    %b = constant 3.14 : f32
    %r = arc.make_struct(%a, %b : i32, f32) : !arc.struct<a : i32, b : f32>
    return %r : !arc.struct<a : i32, b : f32>
  }

  func @ok1() -> !arc.struct<a : i32> {
    %a = constant 4 : i32
    %r = arc.make_struct(%a : i32) : !arc.struct<a : i32>
    return %r : !arc.struct<a : i32>
  }

  func @ok2() -> !arc.struct<a : i32, b : !arc.struct<a : i32> > {
    %a = constant 4 : i32
    %b = constant 3 : i32
    %s = arc.make_struct(%b : i32) : !arc.struct<a : i32>
    %r = arc.make_struct(%a, %s : i32, !arc.struct<a : i32>) : !arc.struct<a : i32, b : !arc.struct<a : i32>>
    return %r : !arc.struct<a : i32, b : !arc.struct<a : i32>>
  }
}
