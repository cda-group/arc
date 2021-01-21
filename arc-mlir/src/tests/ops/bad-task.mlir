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

// -----

module @toplevel {

  func @MyOperator(// Imutables
                   !arc.struct<p0 : f32, p1 : si32>,
		   // State
                   !arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
		               state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>,
		   // Input element
       		   !arc.struct<i : si32, f : f32>,
                   // Ouput stream (repeated)
		   !arc.stream<!arc.struct<i : si32, f : f32>>
		   ) -> ()
      attributes { arc.is_task, arc.in = "In" } {
      ^bb0(%immutables: !arc.struct<p0 : f32, p1 : si32>,
      	   %state: !arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>,
           %input: !arc.struct<i : si32, f : f32>,
	   %output0: !arc.stream<!arc.struct<i : si32, f : f32>>):
	  "arc.emit"(%input, %output0) : (!arc.struct<i : si32, f : f32>, !arc.stream<!arc.struct<i : si32, f : f32>>) -> ()
	  %state1 = "arc.struct_access"(%state) { field = "state1" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.value<!arc.struct<i : si32, f : f32>>
	  %i = arc.constant 4 : si32
	  %f = constant 3.14 : f32
	  %s = arc.make_struct(%f, %i : f32, si32) : !arc.struct<f : f32, i : si32>
	  // expected-error@+2 {{'arc.value_write' op Can't write a value of type '!arc.struct<f : f32, i : si32>' to a state value of type'!arc.struct<i : si32, f : f32>'}}
	   // expected-note@+1 {{see current operation:}}
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<f : f32, i : si32>) -> ()
          return
  }
}

// -----

module @toplevel {

  func @MyOperator(// Imutables
                   !arc.struct<p0 : f32, p1 : si32>,
		   // State
                   !arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
		               state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>,
		   // Input element
       		   !arc.struct<i : si32, f : f32>,
                   // Ouput stream (repeated)
		   !arc.stream<!arc.struct<i : si32, f : f32>>
		   ) -> ()
      attributes { arc.is_task, arc.in = "In" } {
      ^bb0(%immutables: !arc.struct<p0 : f32, p1 : si32>,
      	   %state: !arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>,
           %input: !arc.struct<i : si32, f : f32>,
	   %output0: !arc.stream<!arc.struct<i : si32, f : f32>>):
	  "arc.emit"(%input, %output0) : (!arc.struct<i : si32, f : f32>, !arc.stream<!arc.struct<i : si32, f : f32>>) -> ()
	  %state1 = "arc.struct_access"(%state) { field = "state1" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.value<!arc.struct<i : si32, f : f32>>
          // expected-error@+2 {{'arc.value_read' op Expected result type '!arc.struct<i : si32, f : f32>' not '!arc.struct<f : f32, i : si32>'}}
	   // expected-note@+1 {{see current operation:}}
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<f : f32, i : si32>
          return
  }
}

