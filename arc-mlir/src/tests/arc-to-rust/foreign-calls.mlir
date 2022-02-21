// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustforeigncalls {

  func private @callee_void_void() -> ()

  func private @callee_si32_si32(%in : si32) -> si32

  func private @callee_si32_x2_si32(%a : si32, %b : si32) -> si32

  func private @callee_struct(%in : !arc.struct<foo : si32>)
       -> !arc.struct<foo : si32>

  func @caller0() -> () {
    call @callee_void_void() : () -> ()
    return
  }

  func @caller1(%in : si32) -> (si32) {
    %r = call @callee_si32_si32(%in) : (si32) -> si32
    return %r : si32
  }

  func @caller2(%in0 : si32, %in1 : si32) -> (si32) {
    %r = call @callee_si32_x2_si32(%in0, %in1) : (si32,si32) -> si32
    return %r : si32
  }


  func @caller_struct(%in : !arc.struct<foo : si32>)
       -> !arc.struct<foo : si32> {
    %r = call @callee_struct(%in) : (!arc.struct<foo : si32>)
       	      			     -> !arc.struct<foo : si32>
    return %r : !arc.struct<foo : si32>
  }

}
