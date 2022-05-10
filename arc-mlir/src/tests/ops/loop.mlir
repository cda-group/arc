// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func.func @main() {
    %a = arith.constant 0 : i1
    %b = arith.constant 3.14 : f32
    %c = arith.constant 0.693 : f32
    "arc.if"(%a) ( {
        // expected-error@+2 {{'arc.loop.break' op must be inside a scf.while region}}
    	// expected-note@+1 {{see current operation}}
      "arc.loop.break"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f32) -> ()
    }) : (i1) -> (f32)
    return
  }
}

// -----

module @arctorustloops {

  func.func @a_while_loop(%first : ui64, %limit : ui64, %accum : ui64) -> ui64 {
       %res_cnt, %res_sum = scf.while (%arg1 = %first, %sum = %accum) : (ui64, ui64) -> (ui64, ui64) {
         %condition = arc.cmpi "lt", %arg1, %limit : ui64
         scf.condition(%condition) %arg1, %sum : ui64, ui64
       } do {
         ^bb0(%arg2: ui64, %sum2 : ui64):
	 %one =  arc.constant 1: ui64
         %next = arc.addi %arg2, %one : ui64
         %sum3 = arc.addi %arg2, %sum2 : ui64
	 scf.yield %next, %sum3 : ui64, ui64
       }
       return %res_sum : ui64
  }

}

// -----

module @arctorustloops {

  func.func @a_while_loop_with_a_break_in_before(%first : ui64, %limit : ui64, %accum : ui64) -> ui64 {
       %res_cnt, %res_sum = scf.while (%arg1 = %first, %sum = %accum) : (ui64, ui64) -> (ui64, ui64) {
         %condition = arc.cmpi "lt", %arg1, %limit : ui64

         %a = arc.constant 2 : ui64
         %b = arc.cmpi "ge", %sum, %a : ui64
      	 "arc.if"(%b) ({
	   "arc.loop.break"(%arg1, %sum) : (ui64, ui64) -> ()
         }, {
           "arc.block.result"() : () -> ()
         }) : (i1) -> ()

         scf.condition(%condition) %arg1, %sum : ui64, ui64
       } do {
         ^bb0(%arg2: ui64, %sum2 : ui64):
	 %one =  arc.constant 1: ui64
         %next = arc.addi %arg2, %one : ui64
         %sum3 = arc.addi %arg2, %sum2 : ui64
	 scf.yield %next, %sum3 : ui64, ui64
       }
       return %res_sum : ui64
  }

}

// -----

module @arctorustloops {

  func.func @a_while_loop_with_a_break_in_after(%first : ui64, %limit : ui64, %accum : ui64) -> ui64 {
       %res_cnt, %res_sum = scf.while (%arg1 = %first, %sum = %accum) : (ui64, ui64) -> (ui64, ui64) {
         %condition = arc.cmpi "lt", %arg1, %limit : ui64
         scf.condition(%condition) %arg1, %sum : ui64, ui64
       } do {
         ^bb0(%arg2: ui64, %sum2 : ui64):
	 %one =  arc.constant 1: ui64
         %next = arc.addi %arg2, %one : ui64
         %sum3 = arc.addi %arg2, %sum2 : ui64

         %a = arc.constant 2 : ui64
         %b = arc.cmpi "ge", %sum2, %a : ui64
      	 "arc.if"(%b) ({
	   "arc.loop.break"(%arg2, %sum2) : (ui64, ui64) -> ()
         }, {
           "arc.block.result"() : () -> ()
         }) : (i1) -> ()

	 scf.yield %next, %sum3 : ui64, ui64
       }
       return %res_sum : ui64
  }
}

// -----

module @arctorustloops {

  func.func @a_while_loop_with_a_break_in_after(%first : ui64, %limit : ui64, %accum : ui64) -> ui64 {
       %res_cnt, %res_sum = scf.while (%arg1 = %first, %sum = %accum) : (ui64, ui64) -> (ui64, ui64) {
         %condition = arc.cmpi "lt", %arg1, %limit : ui64
         scf.condition(%condition) %arg1, %sum : ui64, ui64
       } do {
         ^bb0(%arg2: ui64, %sum2 : ui64):
	 %one =  arc.constant 1: ui64
         %next = arc.addi %arg2, %one : ui64
         %sum3 = arc.addi %arg2, %sum2 : ui64

         %a = arc.constant 2 : ui64
         %b = arc.cmpi "ge", %sum2, %a : ui64
      	 "arc.if"(%b) ({
	   // expected-error@+2 {{'arc.loop.break' op returns 1 values parent expects 2}}
    	   // expected-note@+1 {{see current operation}}
	   "arc.loop.break"(%arg2) : (ui64) -> ()
         }, {
           "arc.block.result"() : () -> ()
         }) : (i1) -> ()

	 scf.yield %next, %sum3 : ui64, ui64
       }
       return %res_sum : ui64
  }
}
