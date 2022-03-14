# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task identity(source): (sink) {
    loop {
        on event in source => sink ! event;
    }
}
# ANCHOR_END: example

def main() {}
