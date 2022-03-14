# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
mod foo {
    mod bar {
        mod baz {
            def qux() = 3
        }
    }
}

def main() = foo::bar::baz::qux()
# ANCHOR_END: example
