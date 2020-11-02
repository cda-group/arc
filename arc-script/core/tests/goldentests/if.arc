fun test() {
  let c = true in
  let x = 3 in
  if c { x } else { x }
}

-- args: --verbose --check file
-- expected stdout:
-- fun test() {
--     (let c: bool = (true):bool in
--     (let x: i32 = (3):i32 in
--     (let x3: i32 = (if (c):bool {
--         (x):i32
--     } else {
--         (x):i32
--     }):i32 in
--     (x3):i32):i32):i32):i32
-- }
