// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/arctorustifs/Cargo.toml

module @arctorustifs {
  func @test_0() -> si32 {
    %0 = arc.constant 65 : si32
    %1 = arc.constant 66 : si32
    %2 = constant 1 : i1
    %3 = "arc.if"(%2) ({
      "arc.block.result"(%0) : (si32) -> ()
    }, {
      "arc.block.result"(%1) : (si32) -> ()
    }) : (i1) -> si32
    return %3 : si32
  }
  func @test_1(%c: i1, %arg0: ui32, %arg1: ui32) -> ui32 {
    %3 = "arc.if"(%c) ({
      "arc.block.result"(%arg0) : (ui32) -> ()
    }, {
      "arc.block.result"(%arg1) : (ui32) -> ()
    }) : (i1) -> ui32
    return %3 : ui32
  }
}
