// Check parsing and that round-tripping works
// RUN: arc-mlir %s | arc-mlir
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {

  func.func @union(%a : !arc.stream<si32, ui32>,
                   %b : !arc.stream<si32, ui32>)
  	    -> !arc.stream<si32, ui32> {
    %out = "arc.union"(%a, %b) : (!arc.stream<si32, ui32>,
    	   		          !arc.stream<si32, ui32>)
				   -> !arc.stream<si32, ui32>
    return %out : !arc.stream<si32, ui32>
  }
}
