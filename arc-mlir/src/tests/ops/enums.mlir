// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {
  func @ok0() -> !arc.enum<a : i32, b : f32> {
    %a = constant 4 : i32
    %r = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32, b : f32>
    return %r : !arc.enum<a : i32, b : f32>
  }

  func @ok1() -> !arc.enum<a : i32, b : f32> {
    %b = constant 3.14 : f32
    %r = arc.make_enum (%b : f32) as "b" : !arc.enum<a : i32, b : f32>
    return %r : !arc.enum<a : i32, b : f32>
  }

  func @ok2() -> !arc.enum<a : i32> {
    %a = constant 4 : i32
    %r = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32>
    return %r : !arc.enum<a : i32>
  }

  func @ok3() -> !arc.enum<a : i32, b : !arc.enum<a : i32> > {
    %a = constant 4 : i32
    %b = constant 3 : i32
    %s = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32>
    %r = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32, b : !arc.enum<a : i32>>
    return %r : !arc.enum<a : i32, b : !arc.enum<a : i32>>
  }

  func @ok4() -> !arc.enum<a : i32, b : !arc.enum<a : i32> > {
    %a = constant 4 : i32
    %b = constant 3 : i32
    %s = arc.make_enum (%a : i32) as "a" : !arc.enum<a : i32>
    %r = arc.make_enum (%s : !arc.enum<a : i32>) as "b" : !arc.enum<a : i32, b : !arc.enum<a : i32>>
    return %r : !arc.enum<a : i32, b : !arc.enum<a : i32>>
  }

}
