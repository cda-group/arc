# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
mod foo {
    mod bar {
        val baz = 3;
        # def qux() = baz
    }
    def qux() = bar::baz
}
def qux() = foo::bar::baz
# ANCHOR_END: example

def main() {}
