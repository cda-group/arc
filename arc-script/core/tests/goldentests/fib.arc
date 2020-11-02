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

-- args: --verbose --check file
-- expected stdout:
-- fun fib(n: i32) {
--     (let x3: i32 = (2):i32 in
--     (let x4: bool = ((n):i32 > (x3):i32):bool in
--     (let x13: i32 = (if (x4):bool {
--         (let x5: i32 = (1):i32 in
--         (let x6: i32 = ((n):i32 - (x5):i32):i32 in
--         (let x7: i32 = ((fib):(i32) -> i32((x6):i32)):i32 in
--         (let x8: i32 = (2):i32 in
--         (let x9: i32 = ((n):i32 - (x8):i32):i32 in
--         (let x10: i32 = ((fib):(i32) -> i32((x9):i32)):i32 in
--         (let x11: i32 = ((x7):i32 + (x10):i32):i32 in
--         (x11):i32):i32):i32):i32):i32):i32):i32):i32
--     } else {
--         (let x12: i32 = (0):i32 in
--         (x12):i32):i32
--     }):i32 in
--     (x13):i32):i32):i32):i32
-- }
-- 
-- fun test() {
--     (let x14: i32 = (5):i32 in
--     (let x15: i32 = ((fib):(i32) -> i32((x14):i32)):i32 in
--     (x15):i32):i32):i32
-- }
