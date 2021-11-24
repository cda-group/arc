# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
mod foo {
    mod bar {
        val baz = 3
        def relative() = baz
        def absolute() = ::foo::bar::baz
    }
    def relative() = bar::baz
    def absolute() = ::foo::bar::baz
}

def relative() = foo::bar::baz
def absolute() = ::foo::bar::baz
# ANCHOR_END: example
