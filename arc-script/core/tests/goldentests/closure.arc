let increment = |i:i32| i + 1 in
let foo = 1 in
increment(foo)

-- args: --verbose --check file
-- expected stdout:
-- (let x5: (i32) -> i32 = (|i: i32| {
--     (let x3: i32 = (1):i32 in
--     (let x4: i32 = ((i):i32 + (x3):i32):i32 in
--     (x4):i32):i32):i32
-- }):(i32) -> i32 in
-- (let foo: i32 = (1):i32 in
-- (let x6: i32 = ((x5):(i32) -> i32((foo):i32)):i32 in
-- (x6):i32):i32):i32):i32
