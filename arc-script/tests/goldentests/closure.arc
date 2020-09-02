increment = |i:i32| { i + 1 }
foo = 1
increment(foo)

# args: --verbose file
# expected stdout:
# (x5: (i32) -> i32 = (|i| {(x3: i32 = (1):i32
# (x4: i32 = ((i):i32 + (x3):i32):i32
# (x4):i32):i32):i32}):(i32) -> i32
# (foo: i32 = (1):i32
# (x6: i32 = (increment((foo):i32)):i32
# (x6):i32):i32):i32):i32
