// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctoruststructs {

  func.func @ok0(%in : !arc.struct<foo : si32>) -> !arc.struct<foo : si32> {
    return %in : !arc.struct<foo : si32>
  }

  func.func @ok1(%in : !arc.struct<foo : si32, bar : f32>) ->
      !arc.struct<foo : si32,bar : f32> {
    return %in : !arc.struct<foo : si32, bar: f32>
  }

  func.func @ok2(%in : !arc.struct<foo : si32, bar : f32, inner_struct : !arc.struct<nested : si32>>) -> () {
    return
  }

  func.func @ok3() -> !arc.struct<a : si32, b : f32> {
    %a = arc.constant 4 : si32
    %b = arith.constant 3.14 : f32
    %r = arc.make_struct(%a, %b : si32, f32) : !arc.struct<a : si32, b : f32>
    return %r : !arc.struct<a : si32, b : f32>
  }

  func.func @ok4() -> !arc.struct<a : si32> {
    %a = arc.constant 4 : si32
    %r = arc.make_struct(%a : si32) : !arc.struct<a : si32>
    return %r : !arc.struct<a : si32>
  }

  func.func @ok5() -> !arc.struct<a : si32, b : !arc.struct<a : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    return %r : !arc.struct<a : si32, b : !arc.struct<a : si32>>
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

  func.func @ok8(%a : !arc.struct<a : si32>, %b : !arc.struct<a : si32>) -> !arc.struct<a : si32> {
    return %a : !arc.struct<a : si32>
  }

  func.func @ok9(%a : !arc.struct<a : si32, b : !arc.struct<a : si32>>) -> !arc.struct<a : si32> {
    %r = "arc.struct_access"(%a) { field = "b" } : (!arc.struct<a : si32, b : !arc.struct<a : si32>>) -> !arc.struct<a : si32>
    return %r : !arc.struct<a : si32>
  }

  func.func @ok10() -> si32 {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r0 = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    %r1 = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>

    return %a : si32
  }

  func.func @ok11(%in : !arc.struct<>) -> !arc.struct<> {
    return %in : !arc.struct<>
  }

  func.func @ok12() -> !arc.struct<> {
    %r = arc.make_struct() : !arc.struct<>
    return %r : !arc.struct<>
  }


}
