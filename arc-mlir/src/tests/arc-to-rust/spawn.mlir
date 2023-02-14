// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir
// RUN: arc-mlir -canonicalize -arc-to-rust %s | FileCheck %s
module @arctorustspawn {

  func.func @id(%in : !arc.stream.source<ui32, si32>,
                %out : !arc.stream.sink<ui32, si32>) -> ()
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
        %x = "arc.receive"(%in) : (!arc.stream.source<ui32, si32>) -> si32
        "arc.send"(%x, %out) : (si32, !arc.stream.sink<ui32, si32>) -> ()
        scf.yield
      }
      return
    }

    func.func @main(%in : !arc.stream.source<ui32, si32>,
                    %out : !arc.stream.sink<ui32, si32>) {
       arc.spawn @id(%in, %out) : (!arc.stream.source<ui32, si32>,
                                   !arc.stream.sink<ui32, si32>) -> ()
// CHECK: "rust.spawn"(%arg0, %arg1) {callee = @id} : (!rust<<i32>>, !rust<<i32>>) -> ()
       return
    }
}
