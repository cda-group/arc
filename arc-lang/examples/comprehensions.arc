# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test(stream) {
    [event+1; for event in stream; if event != 0]
}
# ANCHOR_END: example

def main() {}
