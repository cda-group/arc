c = true
x = 3
if c
  then x
  else x

# args: file
# expected stdout:
# c: bool = true
# x: i32 = 3
# x2: i32 = if c
#   then
#     x
#   else
#     x
# x2
