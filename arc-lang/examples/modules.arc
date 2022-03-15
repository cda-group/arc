# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

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
