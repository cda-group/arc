# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

task read_stream(): Stream[i32];

def main() {}

# ANCHOR: pipeline
task map(i, f): (o) {
    for x in i {
        o ! f(receive x)
    }
}

task filter(i, f): (o) {
    for x in i {
        if f(x) {
            o ! x
        }
    }
}

def pipeline_parallel() {
    val s0 = read_stream();
    val s1 = map(s0, |x| x * 2);
    val s2 = filter(s1, |x| x % 2 == 0);
    # ...
}
# ANCHOR_END: pipeline

# ANCHOR: task
def task_parallel() {
    val s0 = source(0..100);
    val s1 = map(s0, |x| x * 2);
    val s2 = filter(s0, |x| x % 2 == 0);
    # ...
}
# ANCHOR_END: task
