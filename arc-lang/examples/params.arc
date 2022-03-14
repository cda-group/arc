# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test(typed_param: i32, untyped_param) = typed_param + untyped_param
# ANCHOR_END: example

def main() {}
