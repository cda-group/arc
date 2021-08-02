// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @arctorustadt {

func @ok0(%in : !arc.adt<"i32">) -> () {
    return
  }

  func @ok2(%in : !arc.adt<"i32">) -> !arc.adt<"i32"> {
    return %in : !arc.adt<"i32">
  }

  func @ok3(%pair : !arc.adt<"(i32, bool)">) -> !arc.adt<"(i32, bool)"> {
    return %pair : !arc.adt<"(i32, bool)">
  }

}
