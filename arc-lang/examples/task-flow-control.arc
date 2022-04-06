# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task producer(): (o) {
    for x in 0.. {
        o ! x;
    }
}

task consumer(i): () {
    for x in i.. {
        sleep(1s);
        print(i);
    }
}

def main() {
    val s = producer();
    consumer(s);
}
# ANCHOR_END: example
