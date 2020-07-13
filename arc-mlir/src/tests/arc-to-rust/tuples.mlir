// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {

  func @makemeatuple() -> tuple<si32,si32> {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %tuple = "arc.make_tuple"(%a, %b) : (si32, si32) -> tuple<si32,si32>
    return %tuple : tuple<si32,si32>
  }

  func @bool_tuple() -> i1 {
    %a = constant 0 : i1
    %b = constant 1 : i1

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
}
