// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {

  func.func @union(%a : !arc.stream<ui32, ui32>,
                   %b : !arc.stream<si32, ui32>)
  	    -> !arc.stream<si32, ui32> {
// expected-error@+2 {{'arc.union' op both input streams should have the same type}}
// expected-note@+1 {{see current operation}}
    %out = "arc.union"(%a, %b) : (!arc.stream<ui32, ui32>,
    	   		          !arc.stream<si32, ui32>)
				   -> !arc.stream<si32, ui32>
    return %out : !arc.stream<si32, ui32>
  }
}

// -----

module @toplevel {

  func.func @union(%a : !arc.stream<si32, si32>,
                   %b : !arc.stream<si32, ui32>)
  	    -> !arc.stream<si32, ui32> {
// expected-error@+2 {{'arc.union' op both input streams should have the same type}}
// expected-note@+1 {{see current operation}}
    %out = "arc.union"(%a, %b) : (!arc.stream<si32, si32>,
    	   		          !arc.stream<si32, ui32>)
				   -> !arc.stream<si32, ui32>
    return %out : !arc.stream<si32, ui32>
  }
}

// -----

module @toplevel {

  func.func @union(%a : !arc.stream<si32, ui32>,
                   %b : !arc.stream<si32, ui32>)
  	    -> !arc.stream<si32, si32> {
// expected-error@+2 {{'arc.union' op input and output streams should have the same type}}
// expected-note@+1 {{see current operation}}
    %out = "arc.union"(%a, %b) : (!arc.stream<si32, ui32>,
    	   		          !arc.stream<si32, ui32>)
				   -> !arc.stream<si32, si32>
    return %out : !arc.stream<si32, si32>
  }
}

// -----

module @toplevel {

  func.func @union(%a : !arc.stream<si32, ui32>,
                   %b : !arc.stream<si32, ui32>)
  	    -> !arc.stream<ui32, ui32> {
// expected-error@+2 {{'arc.union' op input and output streams should have the same type}}
// expected-note@+1 {{see current operation}}
    %out = "arc.union"(%a, %b) : (!arc.stream<si32, ui32>,
    	   		          !arc.stream<si32, ui32>)
				   -> !arc.stream<ui32, ui32>
    return %out : !arc.stream<ui32, ui32>
  }
}
