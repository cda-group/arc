# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust:"source"}
extern def source(): PullChan[i32];

# ANCHOR: example
task duplicate(i): (o1, o2) {
    loop {
        val x = i.pull();
        o1.push(x);
        o2.push(x);
    }
}

def main() {
    val s0 = source();
    val (s1, s2) = duplicate(s0);
}
# ANCHOR_END: example
