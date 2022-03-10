# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

extern def main(
# ANCHOR: record
#{x:i32, y:str} # Record-type
# ANCHOR_END: record
,
# ANCHOR: tuple
(i32, str)      # Tuple-type
# ANCHOR_END: tuple
,
# ANCHOR: function
fun(i32): i32   # Function-type
# ANCHOR_END: function
,
# ANCHOR: exclusive_range
i32..i32        # Exclusive Range-type
# ANCHOR_END: exclusive_range
,
# ANCHOR: inclusive_range
i32..=i32       # Inclusive Range-type    
# ANCHOR_END: inclusive_range
);
