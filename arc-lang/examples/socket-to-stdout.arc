# Run with:
# Terminal 1: $ nc -l 9000
# Terminal 2: $ echo "1\n2\n3" | cargo run -- socket-to-stdout.arc

def f(x) = x + 1;

val i = tcp_reader(socket("127.0.0.1:9000"));
val o = stdout_writer();

source::[i32](i, csv(','), ingestion(1s))
  .map(f)
  .sink(o, csv(','))
  .run();
