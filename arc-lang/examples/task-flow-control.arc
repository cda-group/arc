# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust:"sleep"}
extern def sleep(i32);

# ANCHOR: example
task producer(): (o) {
    loop {
        o.push("Hello");
    }
}

task consumer(i): () {
    loop {
        val x = i.pull();
        sleep(1s);
        print(x);
    }
}

def main() {
    val s = producer();
    consumer(s);
}
# ANCHOR_END: example
