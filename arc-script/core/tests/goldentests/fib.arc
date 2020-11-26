-- RUN: arc-script --mlir --check file %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

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
