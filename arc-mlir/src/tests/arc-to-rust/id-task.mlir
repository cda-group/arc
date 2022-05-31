// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {
  func.func private @pull(!arc.generic_adt<"PullChan", si32>) -> si32 attributes { rust.async, rust.annotation="#[rewrite(unmangled = \"PullChan_pull\")]" }
  func.func private @push(!arc.generic_adt<"PushChan", si32>, si32) -> () attributes { rust.async, rust.annotation="#[rewrite(unmangled = \"PushChan_push\")]" }

  func.func @id(%in : !arc.generic_adt<"PullChan", si32>,
                %out : !arc.generic_adt<"PushChan", si32>) -> ()
    attributes {
      arc.is_task,
      rust.async,
      rust.annotation = "#[rewrite(ephemeral)]"
    } {
      scf.while () : () -> () {
        %condition = arith.constant 1 : i1
        scf.condition(%condition)
      } do {
        ^bb0():
        %x = func.call @pull(%in) : (!arc.generic_adt<"PullChan", si32>) -> si32
        func.call @push(%out, %x) : (!arc.generic_adt<"PushChan", si32>, si32) -> ()
        scf.yield
      }
      return
    }
}
