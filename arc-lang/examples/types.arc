# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def test(
a:
# ANCHOR: record
#{x:i32, y:str} # Record-type
# ANCHOR_END: record
,
b:
# ANCHOR: tuple
(i32, str)      # Tuple-type
# ANCHOR_END: tuple
,
c:
# ANCHOR: function
fun(i32): i32   # Function-type
# ANCHOR_END: function
,
d:
# ANCHOR: arrays
[i32]           # Array-type
# ANCHOR_END: arrays
) = {}

def main() = {}
