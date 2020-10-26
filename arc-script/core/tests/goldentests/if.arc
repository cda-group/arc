let c = true in
let x = 3 in
if c { x } else { x }

-- args: --verbose --check file
-- expected stdout:
-- (let c: bool = (true):bool in
-- (let x: i32 = (3):i32 in
-- (let x2: i32 = (if (c):bool {
--     (x):i32
-- } else {
--     (x):i32
-- }):i32 in
-- (x2):i32):i32):i32):i32
