fun max(a: i32, b: i32) {
    let c = a > b in
    if c { a } else { b }
}

fun test() {
  max(1,2)
}

-- args: --verbose --check file
-- expected stdout:
-- fun max(a: i32, b: i32) {
--     (let x5: bool = ((a):i32 > (b):i32):bool in
--     (let x6: i32 = (if (x5):bool {
--         (a):i32
--     } else {
--         (b):i32
--     }):i32 in
--     (x6):i32):i32):i32
-- }
-- 
-- fun test() {
--     (let x7: i32 = (1):i32 in
--     (let x8: i32 = (2):i32 in
--     (let x9: i32 = ((max):(i32, i32) -> i32((x7):i32, (x8):i32)):i32 in
--     (x9):i32):i32):i32):i32
-- }
