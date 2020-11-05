fun fib(n: i32) {
    if n > 2 {
        fib(n-1) + fib(n-2)
    } else {
        0
    }
}

fun test() {
  fib(5)
}

--[ARC] args: --verbose --check file
--[ARC] expected stdout:
--[ARC] fun fib(n: i32) {
--[ARC]     (let x3: i32 = (2):i32 in
--[ARC]     (let x4: bool = ((n):i32 > (x3):i32):bool in
--[ARC]     (let x13: i32 = (if (x4):bool {
--[ARC]         (let x5: i32 = (1):i32 in
--[ARC]         (let x6: i32 = ((n):i32 - (x5):i32):i32 in
--[ARC]         (let x7: i32 = ((fib):(i32) -> i32((x6):i32)):i32 in
--[ARC]         (let x8: i32 = (2):i32 in
--[ARC]         (let x9: i32 = ((n):i32 - (x8):i32):i32 in
--[ARC]         (let x10: i32 = ((fib):(i32) -> i32((x9):i32)):i32 in
--[ARC]         (let x11: i32 = ((x7):i32 + (x10):i32):i32 in
--[ARC]         (x11):i32):i32):i32):i32):i32):i32):i32):i32
--[ARC]     } else {
--[ARC]         (let x12: i32 = (0):i32 in
--[ARC]         (x12):i32):i32
--[ARC]     }):i32 in
--[ARC]     (x13):i32):i32):i32):i32
--[ARC] }
--[ARC] 
--[ARC] fun test() {
--[ARC]     (let x14: i32 = (5):i32 in
--[ARC]     (let x15: i32 = ((fib):(i32) -> i32((x14):i32)):i32 in
--[ARC]     (x15):i32):i32):i32
--[ARC] }

--[MLIR] args: --mlir --check file
--[MLIR] expected stdout:
--[MLIR] func @x_1(i32) -> (i32) {
--[MLIR]     %x_3 = "arc.constant"() { value = 2 : i32 }: () -> i32
--[MLIR]     %x_4 = "arc.cmpi "gt"  (%x_0,%x_3) : (i32,i32) -> i1
--[MLIR]     %x_13 = "arc.if"(%x_4) ({
--[MLIR]         %x_5 = "arc.constant"() { value = 1 : i32 }: () -> i32
--[MLIR]         %x_6 = "arc.subi"(%x_0,%x_5) : (i32,i32) -> i32
--[MLIR]         %x_7 = call @x_1(%x_6) (i32) -> i32
--[MLIR]         %x_8 = "arc.constant"() { value = 2 : i32 }: () -> i32
--[MLIR]         %x_9 = "arc.subi"(%x_0,%x_8) : (i32,i32) -> i32
--[MLIR]         %x_10 = call @x_1(%x_9) (i32) -> i32
--[MLIR]         %x_11 = "arc.addi"(%x_7,%x_10) : (i32,i32) -> i32
--[MLIR]         "arc.yield"(%x_11) : (i32) -> ()
--[MLIR]     },{
--[MLIR]         %x_12 = "arc.constant"() { value = 0 : i32 }: () -> i32
--[MLIR]         "arc.yield"(%x_12) : (i32) -> ()
--[MLIR]     }) : (i1) -> i32
--[MLIR]     "std.return"(%x_13) : (i32) -> ()
--[MLIR] }
--[MLIR] func @x_2() -> (i32) {
--[MLIR]     %x_14 = "arc.constant"() { value = 5 : i32 }: () -> i32
--[MLIR]     %x_15 = call @x_1(%x_14) (i32) -> i32
--[MLIR]     "std.return"(%x_15) : (i32) -> ()
--[MLIR] }

