// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize
// RUN: arc-mlir-rust-test %t-task %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize -restartable-task -canonicalize -remove-scf-only-tasks -to-scf-only-tasks

module @toplevel {

  func.func @add_one_if_odd(%in : si32) -> si32 attributes { rust.declare } {
    %one = arc.constant 1 : si32
    %bit0 = arc.and %in, %one : si32
    %cond = arc.cmpi "eq", %bit0, %one : si32
    %r = "arc.if"(%cond) ({
      %tmp = arc.addi %one, %in : si32
      "arc.block.result"(%tmp) : (si32) -> ()
    }, {
      "arc.block.result"(%in) : (si32) -> ()
    }) : (i1) -> si32
    return %r : si32
  }

  func.func @id(%in : !arc.stream.source<si32>,
                %out : !arc.stream.sink<si32>) -> ()
    attributes {
      arc.is_task,
      rust.async,
      rust.annotation = "#[rewrite]",
      rust.annotation_task_body = "#[rewrite]"
    } {
      scf.while () : () -> () {
        %condition = arith.constant 1 : i1
        scf.condition(%condition)
      } do {
        ^bb0():
        %x = "arc.receive"(%in) : (!arc.stream.source<si32>) -> si32
        %r = func.call @add_one_if_odd(%x) : (si32) -> si32
        "arc.send"(%r, %out) : (si32, !arc.stream.sink<si32>) -> ()
        scf.yield
      }
      return
    }
}
