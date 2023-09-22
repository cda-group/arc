# XFAIL: *
# RUN: arc-lang %s

type Item = {name: Text, price: f64, mass: f64};
type Order = {id: u32, customer_id: u32, price: f64, items: Vec[Item], time: Time};

val s0: Stream[Order] = source(stdin(), json());
val s1: Stream[Order] = source(file("logs/*.csv"), csv());
val s2: Stream[Order] = source(http("127.0.0.1", 8080, "orders"), tsv());
val s3: Stream[Order] = source(tcp("127.0.0.1", 8081), text());
val s4: Stream[Order] = source(kafka("127.0.0.1", 8082, "orders"), json());

# Can either generate or extract timestamps
val s5: Stream[Order] = source(stdin(), csv(), ingestion());
val s6: Stream[Order] = source(stdin(), csv(), event(fun(e) = e.time));
val s7: Stream[Order] = merge([s0, s1, s2, s3, s4, s5, s6]);

sink(s7, stdout(), json());
sink(s7, file("output.csv"), csv());

def to_usd(price) = price * 1.1;

val s1: Stream[Order] = filter(s0, fun(o) = o.price > 10.0 and o.items.len() > 0);
val s2: Stream[Item] = flatmap(s1, fun(o) = o.items);
val s3: Stream[Text] = map(s2, fun(i) = {name: i.name, price: to_usd(i.price)});
sink(s3, stdout(), json());

source::[Order](stdin(), json())
    .filter(fun(o) = o.price > 10.0 and o.items.len() > 0)
    .flatmap(_.items)
    .map(fun(i) = {i.name, price: i.price.to_usd()})
    .sink(stdout(), json());

from o:Order in source(stdin(), json())
where o.price > 10.0 and o.items.len() > 0
from i in o.items
select {i.name, usd: i.price.to_usd()}
into sink(stdout(), json());

val stream =
    from o:Order in source(stdin(), json())
    where o.price > 10.0 and o.items.len() > 0
    from i in o.items
    select {i.name, usd: i.price.to_usd()};

def query(stream) =
    from o in stream
    where o.price > 10.0 and o.items.len() > 0
    from i in o.items
    select {i.name, usd: i.price.to_usd()};

val stream = query(source::[Order](stdin(), json()));
