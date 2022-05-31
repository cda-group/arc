# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust: "source"}
extern def source(): PullChan[i32];

def main() {}

# ANCHOR: pipeline
task map(i, f): (o) = loop {
    o.push(f(i.pull()));
}

task filter(i, f): (o) = loop {
    val x = i.pull();
    if f(x) {
        o.push(x);
    }
}

def pipeline_parallel() {
    val s0 = source();
    val s1 = map(s0, _ * 2);
    val s2 = filter(s1, _ % 2 == 0);
    # ...
}
# ANCHOR_END: pipeline

# ANCHOR: task
def task_parallel() {
    val s0 = source();
    val s1 = map(s0, _ * 2);
    val s2 = filter(s0, _ % 2 == 0);
    # ...
}
# ANCHOR_END: task
