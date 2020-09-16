fun max(a: i32, b: i32) {
    let c = a > b;
    if c { a } else { b }
}

max(1,2)

# args: --verbose file
# expected stdout:
# fun max(a: i32, b: i32) {
#     ((let x7: bool = ((a):i32 > (b):i32):bool):();
#     ((let x8: i32 = (if (x7):bool {
#         (a):i32
#     } else {
#         (b):i32
#     }):i32):();
#     (x8):i32):i32):i32
# }
# ((let x4: i32 = (1):i32):();
# ((let x5: i32 = (2):i32):();
# ((let x6: i32 = ((max):(i32, i32) -> i32((x4):i32, (x5):i32)):i32):();
# (x6):i32):i32):i32):i32
