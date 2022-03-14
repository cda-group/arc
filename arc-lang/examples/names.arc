# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def this_is_a_name(this_is_also_a_name) {
    val this_is_yet_another_name = 1;
}
# ANCHOR_END: example

def main() {}
