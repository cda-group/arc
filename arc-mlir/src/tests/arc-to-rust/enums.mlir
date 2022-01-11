// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {
  func @ok0() -> !arc.enum<a : si32, b : f32> {
    %a = arc.constant 4 : si32
    %r = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32, b : f32>
    return %r : !arc.enum<a : si32, b : f32>
  }

  func @ok1() -> !arc.enum<a : si32, b : f32> {
    %b = arith.constant 3.14 : f32
    %r = arc.make_enum (%b : f32) as "b" : !arc.enum<a : si32, b : f32>
    return %r : !arc.enum<a : si32, b : f32>
  }

  func @ok2() -> !arc.enum<a : si32> {
    %a = arc.constant 4 : si32
    %r = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32>
    return %r : !arc.enum<a : si32>
  }

  func @ok3() -> !arc.enum<a : si32, b : !arc.enum<a : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32>
    %r = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32, b : !arc.enum<a : si32>>
    return %r : !arc.enum<a : si32, b : !arc.enum<a : si32>>
  }

  func @ok4() -> !arc.enum<a : si32, b : !arc.enum<a : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32>
    %r = arc.make_enum (%s : !arc.enum<a : si32>) as "b" : !arc.enum<a : si32, b : !arc.enum<a : si32>>
    return %r : !arc.enum<a : si32, b : !arc.enum<a : si32>>
  }

  func @ok5() -> !arc.enum<no_value : none> {
    %r = arc.make_enum () as "no_value" : !arc.enum<no_value : none>
    return %r : !arc.enum<no_value : none>
  }

  func @ok6() -> !arc.enum<no_value : none, b : f32> {
    %r = arc.make_enum () as "no_value" : !arc.enum<no_value : none, b : f32>
    return %r : !arc.enum<no_value : none, b : f32>
  }

  func @access0(%e : !arc.enum<a : si32, b : f32>) -> si32 {
    %r = arc.enum_access "a" in (%e : !arc.enum<a : si32, b : f32>) : si32
    return %r : si32
  }

  func @access1(%e : !arc.enum<a : si32, b : f32>) -> f32 {
    %r = arc.enum_access "b" in (%e : !arc.enum<a : si32, b : f32>) : f32
    return %r : f32
  }

  func @check0(%e : !arc.enum<a : si32, b : f32>) -> i1 {
    %r = arc.enum_check (%e : !arc.enum<a : si32, b : f32>) is "a" : i1
    return %r : i1
  }
}
