# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def foo() = 3

def bar() {
    3
}
# ANCHOR_END: example

def main() {}
