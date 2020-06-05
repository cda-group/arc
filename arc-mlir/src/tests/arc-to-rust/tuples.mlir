// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {

  func @makemeatuple() -> tuple<si32,si32> {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %tuple = "arc.make_tuple"(%a, %b) : (si32, si32) -> tuple<si32,si32>
    return %tuple : tuple<si32,si32>
  }
}
