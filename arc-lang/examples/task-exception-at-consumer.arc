# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task producer(xs): (o) {
    for x in xs {
        o ! x;
    }
}

task consumer(i) {
    loop {
        try {
            print(receive i);
        } catch Exception::Receive {
            print("Producer has terminated, therefore I will terminate.");
            return;
        }
    }
}

def main() {
    val stream = producer([1,2,3]);
    consumer(stream);
}
# ANCHOR_END: example
