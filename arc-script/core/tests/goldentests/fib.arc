fun fib(n: i32) {
    if n > 2 {
        fib(n-1) + fib(n-2)
    } else {
        0
    }
}

fib(5)

-- args: --verbose --check file
-- expected stdout:
-- fun fib(n: i32) {
--     ((let x4: i32 = (2):i32):();
--     ((let x5: bool = ((n):i32 > (x4):i32):bool):();
--     ((let x14: i32 = (if (x5):bool {
--         ((let x6: i32 = (1):i32):();
--         ((let x7: i32 = ((n):i32 - (x6):i32):i32):();
--         ((let x8: i32 = ((fib):(i32) -> i32((x7):i32)):i32):();
--         ((let x9: i32 = (2):i32):();
--         ((let x10: i32 = ((n):i32 - (x9):i32):i32):();
--         ((let x11: i32 = ((fib):(i32) -> i32((x10):i32)):i32):();
--         ((let x12: i32 = ((x8):i32 + (x11):i32):i32):();
--         (x12):i32):i32):i32):i32):i32):i32):i32):i32
--     } else {
--         ((let x13: i32 = (0):i32):();
--         (x13):i32):i32
--     }):i32):();
--     (x14):i32):i32):i32):i32
-- }
-- ((let x2: i32 = (5):i32):();
-- ((let x3: i32 = ((fib):(i32) -> i32((x2):i32)):i32):();
-- (x3):i32):i32):i32
