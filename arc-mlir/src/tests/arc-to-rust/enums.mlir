// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @toplevel {
  func @ok0() -> !arc.enum<a0 : si32, b0 : f32> {
    %a = arc.constant 4 : si32
    %r = arc.make_enum (%a : si32) as "a0" : !arc.enum<a0 : si32, b0 : f32>
    return %r : !arc.enum<a0 : si32, b0 : f32>
  }

  func @ok1() -> !arc.enum<a1 : si32, b1 : f32> {
    %b = constant 3.14 : f32
    %r = arc.make_enum (%b : f32) as "b1" : !arc.enum<a1 : si32, b1 : f32>
    return %r : !arc.enum<a1 : si32, b1 : f32>
  }

  func @ok2() -> !arc.enum<a2 : si32> {
    %a = arc.constant 4 : si32
    %r = arc.make_enum (%a : si32) as "a2" : !arc.enum<a2 : si32>
    return %r : !arc.enum<a2 : si32>
  }

  func @ok3() -> !arc.enum<a3 : si32, b3 : !arc.enum<a33 : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_enum (%a : si32) as "a33" : !arc.enum<a33 : si32>
    %r = arc.make_enum (%a : si32) as "a3" : !arc.enum<a3 : si32, b3 : !arc.enum<a33 : si32>>
    return %r : !arc.enum<a3 : si32, b3 : !arc.enum<a33 : si32>>
  }

  func @ok4() -> !arc.enum<a4 : si32, b4 : !arc.enum<a44 : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_enum (%a : si32) as "a44" : !arc.enum<a44 : si32>
    %r = arc.make_enum (%s : !arc.enum<a44 : si32>) as "b4" : !arc.enum<a4 : si32, b4 : !arc.enum<a44 : si32>>
    return %r : !arc.enum<a4 : si32, b4 : !arc.enum<a44 : si32>>
  }

  func @access0(%e : !arc.enum<a5 : si32, b5 : f32>) -> si32 {
    %r = arc.enum_access "a5" in (%e : !arc.enum<a5 : si32, b5 : f32>) : si32
    return %r : si32
  }

  func @access1(%e : !arc.enum<a6 : si32, b6 : f32>) -> f32 {
    %r = arc.enum_access "b6" in (%e : !arc.enum<a6 : si32, b6 : f32>) : f32
    return %r : f32
  }

  func @check0(%e : !arc.enum<a7 : si32, b7 : f32>) -> i1 {
    %r = arc.enum_check (%e : !arc.enum<a7 : si32, b7 : f32>) is "a7" : i1
    return %r : i1
  }
}
