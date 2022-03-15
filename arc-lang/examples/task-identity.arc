# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task identity(source): (sink) {
    loop {
        on event in source => sink ! event;
    }
}
# ANCHOR_END: example

def main() {}
