# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust: "Kafka_source"}
extern def kafka_source[T](String): PullChan[T];

@{rust: "Kafka_sink"}
extern def kafka_sink[T](PullChan[T], String);

# ANCHOR: example
task map(i, f): (o) = loop {
    o.push(f(i.pull()));
}

def main() = {
    val celcius = kafka_source("sensor.celcius");
    val farenheit = map(celcius, _ * 18/10 + 32);
    kafka_sink(farenheit, "sensor.farenheit");
}
# ANCHOR_END: example
