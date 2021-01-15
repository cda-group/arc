// RUN: arc-mlir %s -split-input-file -verify-diagnostics
module @toplevel {

  func @MyOperator(// Imutables
                   !arc.struct<dummy : si32>,
		   // State
                   !arc.struct<dummy : si32>,
		   // Input element
       		   f32,
                   // Ouput stream (repeated)
		   !arc.stream<si32>
		   ) -> ()
      attributes { arc.is_task, arc.in = "In" } {
      ^bb0(%immutables: !arc.struct<dummy : si32>,
      	   %state: !arc.struct<dummy : si32>,
           %input: f32,
	   %output0: !arc.stream<si32>):
	   // expected-error@+2 {{'arc.emit' op Can't emit element of type 'f32' on stream of 'si32'}}
	   // expected-note@+1 {{see current operation:}}
	  "arc.emit"(%input, %output0) : (f32, !arc.stream<si32>) -> () // Should be an error
          return
  }
}

