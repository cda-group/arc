// RUN: arc-mlir -rustcratename arctorustnestedtuplesandstructs -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/arctorustnestedtuplesandstructs/Cargo.toml
// RUN: arc-mlir -rustcratename arctorustnestedtuplesandstructscanon -canonicalize -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/arctorustnestedtuplesandstructscanon/Cargo.toml

module @arctorustnestedtuplesandstructs {

  func @nested0(%in : tuple<si32, tuple<si32, !arc.struct<a : tuple<si32, !arc.struct<b : si32>>>>>) -> si32 {
    %r = arc.constant 4 : si32
    return %r : si32
  }

  func @nested1(%in : !arc.struct<c : tuple<si32, tuple<!arc.struct<d : tuple<si32,si32>>>>>) -> si32 {
    %r = arc.constant 4 : si32
    return %r : si32
  }

}

