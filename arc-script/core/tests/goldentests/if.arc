fun test() {
  let c = true in
  let x = 3 in
  if c { x } else { x }
}

--[ARC] args: --verbose --check file
--[ARC] expected stdout:
--[ARC] fun test() {
--[ARC]     (let c: bool = (true):bool in
--[ARC]     (let x: i32 = (3):i32 in
--[ARC]     (let x3: i32 = (if (c):bool {
--[ARC]         (x):i32
--[ARC]     } else {
--[ARC]         (x):i32
--[ARC]     }):i32 in
--[ARC]     (x3):i32):i32):i32):i32
--[ARC] }

--[MLIR] args: --mlir --check file
--[MLIR] expected stdout:
--[MLIR] func @x_0() -> (i32) {
--[MLIR]     %x_1 = "arc.constant"() { value = true : i1 }: () -> i1
--[MLIR]     %x_2 = "arc.constant"() { value = 3 : i32 }: () -> i32
--[MLIR]     %x_3 = "arc.if"(%x_1) ({
--[MLIR]         "arc.yield"(%x_2) : (i32) -> ()
--[MLIR]     },{
--[MLIR]         "arc.yield"(%x_2) : (i32) -> ()
--[MLIR]     }) : (i1) -> i32
--[MLIR]     "std.return"(%x_3) : (i32) -> ()
--[MLIR] }
