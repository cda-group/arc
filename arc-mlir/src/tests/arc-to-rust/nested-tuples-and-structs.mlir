// RUN: arc-mlir -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {

  func @nested0(%in : tuple<si32, tuple<si32, !arc.struct<a : tuple<si32, !arc.struct<b : si32>>>>>) -> si32 {
    %r = arc.constant 4 : si32
    return %r : si32
  }

  func @nested1(%in : !arc.struct<c : tuple<si32, tuple<!arc.struct<d : tuple<si32,si32>>>>>) -> si32 {
    %r = arc.constant 4 : si32
    return %r : si32
  }

}

