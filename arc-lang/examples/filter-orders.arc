# Run with:
# Terminal 1: $ cargo run -- filter-orders.arc

type Order = {name:String, price:i32, time:Time};

def f(o: Order) = o.price > 100;
def g(o: Order) = {o.name, o.price};

val i = file_reader(path("input/generated/orders.csv"), false);
val o = stdout_writer();

source::[Order](i, csv(','), ingestion(1s))
  .filter(f)
  .map(g)
  .sink(o, csv(','))
  .run();
