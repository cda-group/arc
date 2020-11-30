-- Closures are not yet supported by arc-script
-- XFAIL: *
-- RUN: arc-script --mlir --check file %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun test() {
  let increment = |i:i32| i + 1 in
  let foo = 1 in
  increment(foo)
}

--[ARC] args: --verbose --check file
--[ARC] expected stdout:
--[ARC] fun test() {
--[ARC]     (let x6: (i32) -> i32 = (|i: i32| {
--[ARC]         (let x4: i32 = (1):i32 in
--[ARC]         (let x5: i32 = ((i):i32 + (x4):i32):i32 in
--[ARC]         (x5):i32):i32):i32
--[ARC]     }):(i32) -> i32 in
--[ARC]     (let foo: i32 = (1):i32 in
--[ARC]     (let x7: i32 = ((x6):(i32) -> i32((foo):i32)):i32 in
--[ARC]     (x7):i32):i32):i32):i32
--[ARC] }
