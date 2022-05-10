// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustadt {

  func.func @ok0(%in : !arc.adt<"i32">) -> () {
    return
  }

  func.func @ok2(%in : !arc.adt<"i32">) -> !arc.adt<"i32"> {
    return %in : !arc.adt<"i32">
  }

  func.func @ok4() -> !arc.adt<"i32"> {
    %out = arc.adt_constant "4711" : !arc.adt<"i32">
    return %out : !arc.adt<"i32">
  }

  func.func @ok6(%in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", ui32>)
     -> !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", ui32> {
     return %in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", ui32>
  }

  func.func @ok7(%in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", f64>>)
    -> !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", f64>> {
    return %in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.generic_adt<"crate::arctorustadt::tests::sharable_Foo::Foo", f64>>
  }

  func.func @ok8(%in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.adt<"i32">>)
    -> !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.adt<"i32">> {
    return %in : !arc.generic_adt<"crate::arctorustadt::tests::sharable_Bar::Bar", ui32, !arc.adt<"i32">>
  }
}
