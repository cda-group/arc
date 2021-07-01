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

}
