// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test canon-%t %s -rustinclude %s.rust-tests -canonicalize

module @toplevel {

  func @callee_void_void() -> () {
    return
  }

  func @callee_si32_si32(%in : si32) -> si32 {
    return %in : si32
  }

  func @callee_si32_x2_si32(%a : si32, %b : si32) -> si32 {
    return %b : si32
  }

  func @callee_struct(%in : !arc.struct<foo : si32>)
       -> !arc.struct<foo : si32> {
    return %in : !arc.struct<foo : si32>
  }

  func @callee_tuple(%in : tuple<si32,si32>) -> tuple<si32,si32> {
    return %in : tuple<si32,si32>
  }

  func @callee_mixed(%in : tuple<si32,si32,!arc.struct<a : si32>>)
       -> tuple<si32,si32,!arc.struct<a : si32>> {
    return %in : tuple<si32,si32,!arc.struct<a : si32>>
  }

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

  func @caller_tuple(%in : tuple<si32,si32>) -> tuple<si32,si32> {
    %r = call @callee_tuple(%in) : (tuple<si32,si32>) -> tuple<si32,si32>
    return %r : tuple<si32,si32>
  }

  func @caller_mixed(%in : tuple<si32,si32,!arc.struct<a : si32>>)
       -> tuple<si32,si32,!arc.struct<a : si32>> {
    %r = call @callee_mixed(%in) : (tuple<si32,si32,!arc.struct<a : si32>>)
       	      			    -> tuple<si32,si32,!arc.struct<a : si32>>
    return %in : tuple<si32,si32,!arc.struct<a : si32>>
  }

  func @indir_call0(%in : tuple<si32,si32,!arc.struct<a : si32>>)
       -> tuple<si32,si32,!arc.struct<a : si32>> {
    %f = constant @caller_mixed : (tuple<si32,si32,!arc.struct<a : si32>>)
         -> tuple<si32,si32,!arc.struct<a : si32>>
    %r = call_indirect %f(%in) : (tuple<si32,si32,!arc.struct<a : si32>>)
         -> tuple<si32,si32,!arc.struct<a : si32>>
    return %r : tuple<si32,si32,!arc.struct<a : si32>>
  }

  func @enumfun(%in : !arc.enum<ea : si32, eb : f32>)
       -> !arc.enum<ea : si32, eb : f32> {
    return %in : !arc.enum<ea : si32, eb : f32>
  }

  func @indir_call1(%in : !arc.enum<ea : si32, eb : f32>)
       -> !arc.enum<ea : si32, eb : f32> {
    %f = constant @enumfun : (!arc.enum<ea : si32, eb : f32>)
         -> !arc.enum<ea : si32, eb : f32>
    %r = call_indirect %f(%in) : (!arc.enum<ea : si32, eb : f32>)
         -> !arc.enum<ea : si32, eb : f32>
    return %r : !arc.enum<ea : si32, eb : f32>
  }

  func @tensorfun(%in : tensor<5xi32>)
       -> tensor<5xi32> {
    return %in : tensor<5xi32>
  }

  func @indir_call2(%in : tensor<5xi32>)
       -> tensor<5xi32> {
    %f = constant @tensorfun : (tensor<5xi32>)
         -> tensor<5xi32>
    %r = call_indirect %f(%in) : (tensor<5xi32>)
         -> tensor<5xi32>
    return %r : tensor<5xi32>
  }

  func private @an_external_fun0(si32) -> f32

  func @call_external0(%in : si32) -> f32 {
    %r = call @an_external_fun0(%in) : (si32) -> f32
    return %r : f32
  }

  func @call_external_indirect0(%in : si32) -> f32 {
    %f = constant @an_external_fun0 : (si32) -> f32
    %r = call_indirect %f(%in) : (si32) -> f32
    return %r : f32
  }

  func private @an_external_fun1() -> ((si32) -> si32)

  func @call_external1(%in : si32) -> si32 {
    %f = call @an_external_fun1() : () -> ((si32) -> si32)
    %r = call_indirect %f(%in) : (si32) -> si32
    return %r : si32
  }

  func @call_external_indirect1(%in : si32) -> si32 {
    %thunk = constant @an_external_fun1 : () -> ((si32) -> si32)
    %f = call_indirect %thunk() : () -> ((si32) -> si32)
    %r = call_indirect %f(%in) : (si32) -> si32
    return %r : si32
  }

  func private @crate_Identity() -> ((!arc.stream<!arc.struct<key: si32, value: si32>>) -> !arc.stream<!arc.struct<key: si32, value: si32>>)

    func @crate_main(%input_0: !arc.stream<!arc.struct<key: si32, value: si32>>) -> !arc.stream<!arc.struct<key: si32, value: si32>> {
        %x_8 = constant @crate_Identity : () -> ((!arc.stream<!arc.struct<key: si32, value: si32>>) -> !arc.stream<!arc.struct<key: si32, value: si32>>)
        %x_9 = call_indirect %x_8() : () -> ((!arc.stream<!arc.struct<key: si32, value: si32>>) -> !arc.stream<!arc.struct<key: si32, value: si32>>)
        %x_A = call_indirect %x_9(%input_0) : (!arc.stream<!arc.struct<key: si32, value: si32>>) -> !arc.stream<!arc.struct<key: si32, value: si32>>
        return %x_A : !arc.stream<!arc.struct<key: si32, value: si32>>
    }
}
