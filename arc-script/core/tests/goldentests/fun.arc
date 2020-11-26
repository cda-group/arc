-- RUN: arc-script --mlir --check file %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun max(a: i32, b: i32) {
    let c = a > b in
    if c { a } else { b }
}

fun test() {
  max(1,2)
}

--[ARC] args: --verbose --check file
--[ARC] expected stdout:
--[ARC] fun max(a: i32, b: i32) {
--[ARC]     (let x5: bool = ((a):i32 > (b):i32):bool in
--[ARC]     (let x6: i32 = (if (x5):bool {
--[ARC]         (a):i32
--[ARC]     } else {
--[ARC]         (b):i32
--[ARC]     }):i32 in
--[ARC]     (x6):i32):i32):i32
--[ARC] }
--[ARC] 
--[ARC] fun test() {
--[ARC]     (let x7: i32 = (1):i32 in
--[ARC]     (let x8: i32 = (2):i32 in
--[ARC]     (let x9: i32 = ((max):(i32, i32) -> i32((x7):i32, (x8):i32)):i32 in
--[ARC]     (x9):i32):i32):i32):i32
--[ARC] }
