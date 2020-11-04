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

--[MLIR] args: --verbose --check file
--[MLIR] expected stdout:
--[MLIR] fun test() {
--[MLIR]     (let x6: (i32) -> i32 = (|i: i32| {
--[MLIR]         (let x4: i32 = (1):i32 in
--[MLIR]         (let x5: i32 = ((i):i32 + (x4):i32):i32 in
--[MLIR]         (x5):i32):i32):i32
--[MLIR]     }):(i32) -> i32 in
--[MLIR]     (let foo: i32 = (1):i32 in
--[MLIR]     (let x7: i32 = ((x6):(i32) -> i32((foo):i32)):i32 in
--[MLIR]     (x7):i32):i32):i32):i32
--[MLIR] }
