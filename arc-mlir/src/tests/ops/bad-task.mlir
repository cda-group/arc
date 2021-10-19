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
	  %f = arith.constant 3.14 : f32
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

// -----

module @toplevel {
  func @FoldFun() -> () {
    return
  }

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
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  // expected-error@+2 {{'arc.appender_fold' op folding function has the wrong number of operands, expected 2, found 0}}
	  // expected-note@+1 {{see current operation:}}
          "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32
          return
  }
}

// -----

module @toplevel {

  func @FoldFun(si32, si32) -> () {
  ^bb0(%a: si32, %b : si32):
    return
  }

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
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  // expected-error@+2 {{'arc.appender_fold' op folding function has to return a single value, found 0 values}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32
          return
  }
}

// -----

module @toplevel {

  func @FoldFun(si32, si32) -> si32 {
  ^bb0(%a: si32, %b : si32):
    %i = arc.constant 4 : si32
    return %i : si32
  }

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
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  // expected-error@+2 {{'arc.appender_fold' op expected type of accumulator initializer to match type of folding function accumulator, found 'si32' expected 'f32'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32
          return
  }
}

// -----

module @toplevel {

  func @FoldFun(f32, si32) -> si32 {
  ^bb0(%a: f32, %b : si32):
    %i = arc.constant 4 : si32
    return %i : si32
  }

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
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  // expected-error@+2 {{'arc.appender_fold' op expected type of folding function accumulator to match folding function result type, found 'si32' expected 'f32'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32
          return
  }
}

// -----

module @toplevel {

  func @FoldFun(f32, si32) -> f32 {
  ^bb0(%a: f32, %b : si32):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
  	  // expected-error@+2 {{'arc.appender_fold' op expected type of folding function input to match appender type, found 'f32' expected 'si32'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32
          return
  }
}

// -----

module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>

	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

	  // expected-error@+2 {{'arc.map_insert' op key type 'f32' does not match map key type 'ui64'}}
	  // expected-note@+1 {{see current operation:}}
          "arc.map_insert"(%state3, %f, %f) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, f32, f32) -> ()

          return
  }
}

// -----
module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
  	  %u = arc.constant 8 : ui64
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>

	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

	  // expected-error@+2 {{'arc.map_insert' op value type 'f32' does not match map value type '!arc.struct<i : si32, f : f32>'}}
	  // expected-note@+1 {{see current operation:}}
          "arc.map_insert"(%state3, %u, %f) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64, f32) -> ()

          return
  }
}

// -----
module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
	  %u = arc.constant 8 : ui64
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

	  // expected-error@+2 {{'arc.map_remove' op key type 'f32' does not match map key type 'ui64'}}
	  // expected-note@+1 {{see current operation:}}
          "arc.map_remove"(%state3, %f) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, f32) -> ()

return
  }
}

// -----
module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
	  %u = arc.constant 8 : ui64
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>
	  // expected-error@+2 {{'arc.map_get' op key type 'f32' does not match map key type 'ui64'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.map_get"(%state3, %f) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, f32) -> (!arc.struct<i : si32, f : f32>)

return
  }
}

// -----
module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
	  %u = arc.constant 8 : ui64
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

	  // expected-error@+2 {{'arc.map_get' op result type 'f32' does not match map value type '!arc.struct<i : si32, f : f32>'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.map_get"(%state3, %u) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64) -> (f32)

return
  }
}

// -----
module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = arith.constant 3.14 : f32
    return %f : f32
  }

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
	  %i = arc.constant 4 : si32
	  %u = arc.constant 8 : ui64
	  %f = arith.constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

	  // expected-error@+2 {{'arc.map_contains' op key type 'f32' does not match map key type 'ui64'}}
	  // expected-note@+1 {{see current operation:}}
	  "arc.map_contains"(%state3, %f) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, f32) -> i1

return
  }
}
