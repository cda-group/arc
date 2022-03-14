# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test(i) {
    task: (o) = loop {
        on x in i => o ! x
    }
}
# ANCHOR_END: example

def main() {}
