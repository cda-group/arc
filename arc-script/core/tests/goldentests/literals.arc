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

-- args: --check file
-- expected stdout:
-- fun test() {
--     let xi8: i8 = 1i8 in
--     let xi16: i16 = 1i16 in
--     let xi32: i32 = 1 in
--     let xi64: i64 = 1i64 in
--     let xf32: f32 = 1.1f32 in
--     let xf64: f64 = 1.1 in
--     let xbool: bool = true in
--     let x8: i32 = 1 in
--     x8
-- }

