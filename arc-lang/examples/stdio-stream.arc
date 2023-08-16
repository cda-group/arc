def f(x) = x > 1;

source::[i32](stdin_reader(), csv(','), ingestion(1s))
  .filter(f)
  .sink(file_writer(path("foo.csv")), csv(','))
  .run();
