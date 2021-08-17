// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @arctorustloops {

  func @a_while_loop(%first : ui64, %limit : ui64, %accum : ui64) -> ui64 {
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
