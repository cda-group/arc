# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task identity(i): (o) = loop { o.push(i.pull()); }
# ANCHOR_END: example

def main() {}
