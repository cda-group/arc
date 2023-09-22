# Run with:
# Terminal 1: $ cargo run -- file-to-file.arc

def f(x) = x + 1;

val i = file_reader(path("input/generated/numbers.csv"), false);
val o = file_writer(path("output/numbers-plus-one.csv"));

source::[i32](i, csv(','), ingestion(1s))
  .map(f)
  .sink(o, csv(','))
  .run();
