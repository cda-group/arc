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

  func @access0(%e : !arc.enum<a : i32, b : f32>) -> i32 {
    %r = arc.enum_access "a" in (%e : !arc.enum<a : i32, b : f32>) : i32
    return %r : i32
  }

  func @access1(%e : !arc.enum<a : i32, b : f32>) -> f32 {
    %r = arc.enum_access "b" in (%e : !arc.enum<a : i32, b : f32>) : f32
    return %r : f32
  }

  func @check0(%e : !arc.enum<a : i32, b : f32>) -> i1 {
    %r = arc.enum_check (%e : !arc.enum<a : i32, b : f32>) is "a" : i1
    return %r : i1
  }

  func @check1(%e : !arc.enum<a : i32, b : f32>) -> i1 {
    %r = arc.enum_check (%e : !arc.enum<a : i32, b : f32>) is "b" : i1
    return %r : i1
  }
}
