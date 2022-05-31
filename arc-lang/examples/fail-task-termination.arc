# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task map_once(i): (o) {
    loop {
        o.push(i.pull());
        break;
    }
}
# ANCHOR_END: example

def main() = 1
