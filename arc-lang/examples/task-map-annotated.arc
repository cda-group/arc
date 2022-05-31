# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust: "Kafka_source"}
extern def kafka_source[T](String): PullChan[T];

@{rust: "Kafka_sink"}
extern def kafka_sink[T](PullChan[T], String);

# ANCHOR: example
task map[A, B](i: PullChan[A], f: fun(A):B): (o: PushChan[B]) {
    loop {
        o.push(f(i.pull()));
    }
}

def main() = {
    val c: PullChan[i32] = kafka_source("sensor.celcius");
    val f: PullChan[i32] = map(c, fun(c: i32) = c * 18 / 10 + 32);
    kafka_sink(f, "sensor.farenheit");
}
# ANCHOR_END: example
