// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {
  func @zero_args() -> ui16 {
    %t = arc.constant 4711 : ui16
    return %t : ui16
  }

  func @one_arg(%a : ui16) -> ui16 {
    return %a : ui16
  }

  func @two_args(%a : ui16, %b : ui16) -> ui16 {
    return %b : ui16
  }
}
