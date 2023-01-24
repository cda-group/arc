// RUN: arc-mlir %s -split-input-file -verify-diagnostics
module @toplevel {

  func.func @MyOperator(// Imutables
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

	  %i = arc.constant 4 : si32
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%f, %i : f32, si32) : !arc.struct<f : f32, i : si32>
	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  // expected-error@+2 {{'arc.appender_push' op can't push a value of type '!arc.struct<f : f32, i : si32>' to an appender expecting type '!arc.struct<i : si32, f : f32>'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<f : f32, i : si32>) -> ()

          return
  }
}

