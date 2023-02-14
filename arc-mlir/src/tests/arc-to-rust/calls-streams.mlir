// XFAIL: *
// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {

  func.func private @crate_Identity() -> ((!arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>) -> !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>)

    func.func @crate_main(%input_0: !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>) -> !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>> {
        %x_8 = constant @crate_Identity : () -> ((!arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>) -> !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>)
        %x_9 = call_indirect %x_8() : () -> ((!arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>) -> !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>)
        %x_A = call_indirect %x_9(%input_0) : (!arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>) -> !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>
        return %x_A : !arc.stream.source<ui32, !arc.struct<key: si32, value: si32>>
    }

}
