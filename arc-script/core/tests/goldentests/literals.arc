-- RUN: arc-script --mlir --check file %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun test() {
  let xi8   = 1i8 in
  let xi16  = 1i16 in
  let xi32  = 1 in
  let xi64  = 1i64 in
  let xf32  = 1.1f32 in
  let xf64  = 1.1 in
  let xbool = true in
  1
}

--[ARC] args: --check file
--[ARC] expected stdout:
--[ARC] fun test() {
--[ARC]     let xi8: i8 = 1i8 in
--[ARC]     let xi16: i16 = 1i16 in
--[ARC]     let xi32: i32 = 1 in
--[ARC]     let xi64: i64 = 1i64 in
--[ARC]     let xf32: f32 = 1.1f32 in
--[ARC]     let xf64: f64 = 1.1 in
--[ARC]     let xbool: bool = true in
--[ARC]     let x8: i32 = 1 in
--[ARC]     x8
--[ARC] }

--[MLIR] args: --mlir --check file
--[MLIR] expected stdout:
--[MLIR] func @x_0() -> (i32) {
--[MLIR]     %x_1 = "arc.constant"() { value = 1 : i8 }: () -> i8
--[MLIR]     %x_2 = "arc.constant"() { value = 1 : i16 }: () -> i16
--[MLIR]     %x_3 = "arc.constant"() { value = 1 : i32 }: () -> i32
--[MLIR]     %x_4 = "arc.constant"() { value = 1 : i64 }: () -> i64
--[MLIR]     %x_5 = "arc.constant"() { value = 1.1 : f32 }: () -> f32
--[MLIR]     %x_6 = "arc.constant"() { value = 1.1 : f64 }: () -> f64
--[MLIR]     %x_7 = "arc.constant"() { value = true : i1 }: () -> i1
--[MLIR]     %x_8 = "arc.constant"() { value = 1 : i32 }: () -> i32
--[MLIR]     "std.return"(%x_8) : (i32) -> ()
--[MLIR] }
