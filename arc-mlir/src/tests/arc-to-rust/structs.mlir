// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {

  func @ok0(%in : !arc.struct<foo : i32>) -> !arc.struct<foo : i32> {
    return %in : !arc.struct<foo : i32>
  }

  func @ok1(%in : !arc.struct<foo : i32, bar : f32>) ->
      !arc.struct<foo : i32,bar : f32> {
    return %in : !arc.struct<foo : i32, bar: f32>
  }

  func @ok2(%in : !arc.struct<foo : i32, bar : f32, inner_struct : !arc.struct<nested : i32>>) -> () {
    return
  }
}
