# Run with:
# Terminal 1: $ cargo run -- socket-to-stdout.arc
# Terminal 2: $ nc 127.0.0.1 9000

def f(x) = x + 1;

val i = stdin_reader();
val o = tcp_writer(socket("127.0.0.1:9000"));

source::[i32](i, csv(','), ingestion(1s))
  .map(f)
  .sink(o, csv(','))
  .run();
