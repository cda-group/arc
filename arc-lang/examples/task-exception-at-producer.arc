# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task producer(xs): (o) {
    for x in xs {
        try {
            o ! x;
        } catch Exception::Emit {
            print("Consumer has terminated, therefore I will terminate.");
            return;
        }
    }
}

task consumer(i) {
    for x in i {
        print(x);
        if rand() % 100 == 0 {
            break;
        }
    }
}

def main() {
    val stream = producer([1,2,3]);
    consumer(stream);
}
# ANCHOR_END: example
