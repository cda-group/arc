// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @arctorustspawn {

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

    func.func @main(%in : !arc.stream.source<si32>,
                    %out : !arc.stream.sink<si32>) {
       arc.spawn @id(%in, %out) : (!arc.stream.source<si32>,
                                   !arc.stream.sink<si32>) -> ()
       return
    }
}
