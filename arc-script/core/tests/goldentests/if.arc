let c = true;
let x = 3;
if c { x } else { x }

-- args: --verbose --check file
-- expected stdout:
-- ((let c: bool = (true):bool):();
-- ((let x: i32 = (3):i32):();
-- ((let x2: i32 = (if (c):bool {
--     (x):i32
-- } else {
--     (x):i32
-- }):i32):();
-- (x2):i32):i32):i32):i32
