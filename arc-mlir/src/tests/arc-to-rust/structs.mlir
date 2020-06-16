// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {

  func @ok0(%in : !arc.struct<foo : si32>) -> !arc.struct<foo : si32> {
    return %in : !arc.struct<foo : si32>
  }

  func @ok1(%in : !arc.struct<foo : si32, bar : f32>) ->
      !arc.struct<foo : si32,bar : f32> {
    return %in : !arc.struct<foo : si32, bar: f32>
  }

  func @ok2(%in : !arc.struct<foo : si32, bar : f32, inner_struct : !arc.struct<nested : si32>>) -> () {
    return
  }

  func @ok3() -> !arc.struct<a : si32, b : f32> {
    %a = arc.constant 4 : si32
    %b = constant 3.14 : f32
    %r = arc.make_struct(%a, %b : si32, f32) : !arc.struct<a : si32, b : f32>
    return %r : !arc.struct<a : si32, b : f32>
  }

  func @ok4() -> !arc.struct<a : si32> {
    %a = arc.constant 4 : si32
    %r = arc.make_struct(%a : si32) : !arc.struct<a : si32>
    return %r : !arc.struct<a : si32>
  }

  func @ok5() -> !arc.struct<a : si32, b : !arc.struct<a : si32> > {
    %a = arc.constant 4 : si32
    %b = arc.constant 3 : si32
    %s = arc.make_struct(%b : si32) : !arc.struct<a : si32>
    %r = arc.make_struct(%a, %s : si32, !arc.struct<a : si32>) : !arc.struct<a : si32, b : !arc.struct<a : si32>>
    return %r : !arc.struct<a : si32, b : !arc.struct<a : si32>>
  }
}
