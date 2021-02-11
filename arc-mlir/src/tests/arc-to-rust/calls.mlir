// RUN: arc-mlir -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

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
}
