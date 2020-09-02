fun max(a: i32, b: i32)
  c = a == b
  if c
    then
      a
    else
      b

max(1,2)

# args: --verbose file
# expected stdout:
# fun max(a: i32, b: i32)
#   (x7: bool = ((a):i32 == (b):i32):bool
#   (x8: i32 = (if (x7):bool
#     then
#       (a):i32
#     else
#       (b):i32):i32
#   (x8):i32):i32):i32
# end
# 
# (x4: i32 = (1):i32
# (x5: i32 = (2):i32
# (x6: i32 = (max((x4):i32, (x5):i32)):i32
# (x6):i32):i32):i32):i32
