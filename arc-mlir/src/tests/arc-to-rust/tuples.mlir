// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @arctorusttuples {

  func @makemeatuple() -> tuple<si32,si32> {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %tuple = "arc.make_tuple"(%a, %b) : (si32, si32) -> tuple<si32,si32>
    return %tuple : tuple<si32,si32>
  }

  func @bool_tuple() -> i1 {
    %a = arith.constant 0 : i1
    %b = arith.constant 1 : i1

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 0 } : (tuple<i1,i1>) -> i1

    return %elem : i1
  }

  func @const_tuple() -> si32 {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %tuple = "arc.make_tuple"(%a, %b) : (si32, si32) -> tuple<si32,si32>

    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<si32,si32>) -> si32

    return %elem : si32
  }

  func @tuple_access(%tuple : tuple<si32,si32>) -> si32 {
    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<si32,si32>) -> si32
    return %elem : si32
  }

  func @make_nested(%inner : tuple<si32,si32>) -> tuple<si32,si32,tuple<si32,si32>> {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %outer = "arc.make_tuple"(%a, %b, %inner) : (si32, si32, tuple<si32,si32>) -> tuple<si32,si32,tuple<si32,si32>>

    return %outer : tuple<si32,si32,tuple<si32,si32>>
  }

  func @make_with_struct(%s : !arc.struct<a : si32>) -> tuple<si32,si32,!arc.struct<a : si32>> {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %outer = "arc.make_tuple"(%a, %b, %s) : (si32, si32, !arc.struct<a : si32>) -> tuple<si32,si32,!arc.struct<a : si32>>

    return %outer : tuple<si32,si32,!arc.struct<a : si32>>
  }

  func @single_element_tuple0() -> tuple<si32> {
    %a = arc.constant 7 : si32
    %r = "arc.make_tuple"(%a) : (si32) -> tuple<si32>
    return %r : tuple<si32>
  }

  func @single_element_tuple1(%in : tuple<si32>) -> si32 {
    %r = "arc.index_tuple"(%in) { index = 0 } : (tuple<si32>) -> si32
    return %r : si32
  }
}
