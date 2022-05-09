// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir
module @toplevel {

  func.func @MyOperator(// Imutables
                   !arc.struct<dummy : si32>,
		   // State
                   !arc.struct<dummy : si32>,
		   // Input element
       		   si32,
                   // Output stream (repeated)
		   !arc.stream<si32>
		   ) -> ()
      attributes { arc.is_task, arc.in = "In" } {
      ^bb0(%immutables: !arc.struct<dummy : si32>,
      	   %state: !arc.struct<dummy : si32>,
           %input: si32,
	   %output0: !arc.stream<si32>):
	  "arc.emit"(%input, %output0) : (si32, !arc.stream<si32>) -> ()
          return
  }
}

