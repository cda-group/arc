// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustifs {
  func.func @test_0() -> si32 {
    %0 = arc.constant 65 : si32
    %1 = arc.constant 66 : si32
    %2 = arith.constant 1 : i1
    %3 = "arc.if"(%2) ({
      "arc.block.result"(%0) : (si32) -> ()
    }, {
      "arc.block.result"(%1) : (si32) -> ()
    }) : (i1) -> si32
    return %3 : si32
  }
  func.func @test_1(%c: i1, %arg0: ui32, %arg1: ui32) -> ui32 {
    %3 = "arc.if"(%c) ({
      "arc.block.result"(%arg0) : (ui32) -> ()
    }, {
      "arc.block.result"(%arg1) : (ui32) -> ()
    }) : (i1) -> ui32
    return %3 : ui32
  }
  func.func @test_2(%c: i1, %arg0: ui32, %arg1: ui32) -> () {
    "arc.if"(%c) ({
      "arc.block.result"() : () -> ()
    }, {
      "arc.block.result"() : () -> ()
    }) : (i1) -> ()
    return
  }

  func.func @test_3(%c: i1, %arg0: ui32, %arg1: ui32) -> ui32 {
    %3 = "arc.if"(%c) ({
      "arc.return"(%arg0) : (ui32) -> ()
    }, {
      "arc.block.result"(%arg1) : (ui32) -> ()
    }) : (i1) -> ui32
    return %3 : ui32
  }
  func.func @test_4(%c: i1, %arg0: ui32, %arg1: ui32) -> ui32 {
    %3 = "arc.if"(%c) ({
      "arc.block.result"(%arg1) : (ui32) -> ()
    }, {
      "arc.return"(%arg0) : (ui32) -> ()
    }) : (i1) -> ui32
    return %3 : ui32
  }
}
