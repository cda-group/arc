// RUN: arc-mlir %s
//!
//! ```txt
//! type MyData = { i:i32, f:f32 };
//!
//! task MyOperator(p0:f32, p1:i32) (In(MyData)) -> (Out(MyData))
//!
//!     -- State types supported by Arcon.
//!     state state1: Value<MyData>;
//!     state state2: Appender<MyData>;
//!     state state3: Map<u64, MyData>;
//!
//!     on In(data) => {
//!         if let Some(_) = state1.get() {
//!             let foo = { i=0 + p0, f=1.1 + p1 };
//!             emit Out(foo);
//!             state1.set(foo);
//!             state2.clear();
//!         } else {
//!             state2.append(data);
//!             state3.put(5, data);
//!             emit Out(data);
//!         }
//!     }
//! end
//! ```

// MyData
//   !arc.struct<i : si32, f : f32>

module @toplevel {

  func @FoldFun(f32, !arc.struct<i : si32, f : f32>) -> f32 {
  ^bb0(%a: f32, %b : !arc.struct<i : si32, f : f32>):
    %f = constant 3.14 : f32
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
	  %u = arc.constant 8 : ui64
	  %f = constant 3.14 : f32
	  %s = arc.make_struct(%i, %f : si32, f32) : !arc.struct<i : si32, f : f32>
	  "arc.value_write"(%state1, %s) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  %sr = "arc.value_read"(%state1) : (!arc.arcon.value<!arc.struct<i : si32, f : f32>>) -> !arc.struct<i : si32, f : f32>

	  %state2 = "arc.struct_access"(%state) { field = "state2" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.appender<!arc.struct<i : si32, f : f32>>
	  "arc.appender_push"(%state2, %s) : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, !arc.struct<i : si32, f : f32>) -> ()
	  "arc.appender_fold"(%state2, %f) {fun=@FoldFun} : (!arc.arcon.appender<!arc.struct<i : si32, f : f32>>, f32) -> f32

	  %state3 = "arc.struct_access"(%state) { field = "state3" } : (!arc.struct<state1 : !arc.arcon.value<!arc.struct<i : si32, f : f32>>,
	                       state2 : !arc.arcon.appender<!arc.struct<i : si32, f : f32>>,
			       state3 : !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>>) -> !arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>

          "arc.map_insert"(%state3, %u, %s) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64, !arc.struct<i : si32, f : f32>) -> ()
          "arc.map_remove"(%state3, %u) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64) -> ()
	  "arc.map_get"(%state3, %u) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64) -> (!arc.struct<i : si32, f : f32>)
  	  "arc.map_contains"(%state3, %u) : (!arc.arcon.map<ui64, !arc.struct<i : si32, f : f32>>, ui64) -> i1

return
  }
}

