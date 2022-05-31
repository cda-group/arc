# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test(i) {
    val map = task(): (o) = loop {
        o.push(i.pull());
    };
}
# ANCHOR_END: example

def main() {}
