# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

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

def main() {}
