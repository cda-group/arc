// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {

  func.func @callee_void_void() -> () attributes { rust.declare } {
    return
  }

  func.func @callee_si32_si32(%in : si32) -> si32 attributes { rust.declare } {
    return %in : si32
  }

  func.func @callee_si32_x2_si32(%a : si32, %b : si32) -> si32 attributes { rust.declare } {
    return %b : si32
  }

  func.func @callee_struct(%in : !arc.struct<foo : si32>)
       -> !arc.struct<foo : si32> attributes { rust.declare } {
    return %in : !arc.struct<foo : si32>
  }

  func.func @caller0() -> () attributes { rust.declare }  {
    call @callee_void_void() : () -> ()
    return
  }

  func.func @caller1(%in : si32) -> (si32) attributes { rust.declare }  {
    %r = call @callee_si32_si32(%in) : (si32) -> si32
    return %r : si32
  }

  func.func @caller2(%in0 : si32, %in1 : si32) -> (si32) attributes { rust.declare }  {
    %r = call @callee_si32_x2_si32(%in0, %in1) : (si32,si32) -> si32
    return %r : si32
  }


  func.func @caller_struct(%in : !arc.struct<foo : si32>)
       -> !arc.struct<foo : si32> {
    %r = call @callee_struct(%in) : (!arc.struct<foo : si32>)
       	      			     -> !arc.struct<foo : si32>
    return %r : !arc.struct<foo : si32>
  }

  func.func @enumfun(%in : !arc.enum<ea : si32, eb : f32>)
       -> !arc.enum<ea : si32, eb : f32> attributes { rust.declare } {
    return %in : !arc.enum<ea : si32, eb : f32>
  }

  func.func @indir_call1(%in : !arc.enum<ea : si32, eb : f32>)
       -> !arc.enum<ea : si32, eb : f32> attributes { rust.declare } {
    %f = constant @enumfun : (!arc.enum<ea : si32, eb : f32>)
         -> !arc.enum<ea : si32, eb : f32>
    %r = call_indirect %f(%in) : (!arc.enum<ea : si32, eb : f32>)
         -> !arc.enum<ea : si32, eb : f32>
    return %r : !arc.enum<ea : si32, eb : f32>
  }

  // func.func @tensorfun(%in : tensor<5xi32>)
  //      -> tensor<5xi32> attributes { rust.declare } {
  //   return %in : tensor<5xi32>
  // }

  // func.func @indir_call2(%in : tensor<5xi32>)
  //      -> tensor<5xi32> attributes { rust.declare } {
  //   %f = constant @tensorfun : (tensor<5xi32>)
  //        -> tensor<5xi32>
  //   %r = call_indirect %f(%in) : (tensor<5xi32>)
  //        -> tensor<5xi32>
  //   return %r : tensor<5xi32>
  // }

  func.func private @an_external_fun0(si32) -> f32 attributes { rust.declare }

  func.func @call_external0(%in : si32) -> f32 attributes { rust.declare } {
    %r = call @an_external_fun0(%in) : (si32) -> f32
    return %r : f32
  }

  func.func @call_external_indirect0(%in : si32) -> f32 attributes { rust.declare } {
    %f = constant @an_external_fun0 : (si32) -> f32
    %r = call_indirect %f(%in) : (si32) -> f32
    return %r : f32
  }

  // TODO: Try to find a way to support returning functions into Arc-Lang
  // func private @an_external_fun1() -> ((si32) -> si32) attributes { rust.declare }

  // func.func @call_external1(%in : si32) -> si32 attributes { rust.declare } {
  //   %f = call @an_external_fun1() : () -> ((si32) -> si32)
  //   %r = call_indirect %f(%in) : (si32) -> si32
  //   return %r : si32
  // }

  // func.func @call_external_indirect1(%in : si32) -> si32 attributes { rust.declare } {
  //   %thunk = constant @an_external_fun1 : () -> ((si32) -> si32)
  //   %f = call_indirect %thunk() : () -> ((si32) -> si32)
  //   %r = call_indirect %f(%in) : (si32) -> si32
  //   return %r : si32
  // }

  func.func private @an_external_fun_with_other_name0(si32) -> si32 attributes { rust.declare, "arc.rust_name" = "the_name_on_the_rust_side" }

  func.func @call_external2(%in : si32) -> si32 attributes { rust.declare } {
    %r = call @an_external_fun_with_other_name0(%in) : (si32) -> si32
    return %r : si32
  }

  func.func private @a_function_called_something_else_in_rust(%in : si32) -> si32
     attributes { "arc.rust_name" = "defined_in_arc" } {
    return %in : si32
  }

  func.func @call_renamed_local(%in : si32) -> si32 attributes { rust.declare } {
    %r = call @a_function_called_something_else_in_rust(%in) : (si32) -> si32
    return %r : si32
  }

  // func private @an_external_fun3(si32) -> si32 attributes { rust.declare }

  // Async functions
  func.func private @an_external_async_fun() -> si32 attributes { rust.async }

  func.func @call_external_async() -> si32 attributes { rust.declare, rust.async } {
    %r = call @an_external_async_fun() : () -> si32
    return %r : si32
  }

}
