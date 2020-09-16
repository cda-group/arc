let increment = |i:i32| i + 1;
let foo = 1;
increment(foo)

# args: --verbose file
# expected stdout:
# ((let x5: (i32) -> i32 = (|i:i32| {
#     ((let x3: i32 = (1):i32):();
#     ((let x4: i32 = ((i):i32 + (x3):i32):i32):();
#     (x4):i32):i32):i32
# }):(i32) -> i32):();
# ((let foo: i32 = (1):i32):();
# ((let x6: i32 = ((x5):(i32) -> i32((foo):i32)):i32):();
# (x6):i32):i32):i32):i32
