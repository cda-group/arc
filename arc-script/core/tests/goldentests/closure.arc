fun test() {
  let increment = |i:i32| i + 1 in
  let foo = 1 in
  increment(foo)
}

-- args: --verbose --check file
-- expected stdout:
-- fun test() {
--     (let x6: (i32) -> i32 = (|i: i32| {
--         (let x4: i32 = (1):i32 in
--         (let x5: i32 = ((i):i32 + (x4):i32):i32 in
--         (x5):i32):i32):i32
--     }):(i32) -> i32 in
--     (let foo: i32 = (1):i32 in
--     (let x7: i32 = ((x6):(i32) -> i32((foo):i32)):i32 in
--     (x7):i32):i32):i32):i32
-- }
