// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @toplevel {
  func.func @id(%in : !arc.stream.source<si32>,
                %out : !arc.stream.sink<si32>) -> ()
    attributes {
      arc.is_task,
      rust.async,
      rust.annotation = "#[rewrite(nonpersistent)]"
    } {
      scf.while () : () -> () {
        %condition = arith.constant 1 : i1
        scf.condition(%condition)
      } do {
        ^bb0():
        %x = "arc.receive"(%in) : (!arc.stream.source<si32>) -> si32
        "arc.send"(%x, %out) : (si32, !arc.stream.sink<si32>) -> ()
        scf.yield
      }
      return
    }
}
