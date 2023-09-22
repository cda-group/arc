+++
+++
```
type Item = {name:String, price:f64, mass:f64};
type Request = {id:u32, items: Vec[Item], t:Time, src_id:u32, dst_id:u32};
type Area = {id:u32, lon:f32, lat:f32};

val model = model(path("cost_predictor.pb"), tf()): Model[[f32;3],[f32;1];
val areas = set(path("areas.csv"), csv()): Set[Area];

---rust
fn custom_predicate(r: Request) -> bool { /* Rust code */ }
---

from r:Request in source(kafka("127.0.0.1:9092", "requests"), json())
where fun(r) = r.items.len() > 0
join a0 in areas on a0.id == r.src_id
join a1 in areas on a1.id == r.dst_id
val features = [r.items.sum_by(_.mass), f32(r.t.day()), distance(a0, a1)]
val [shipment_cost] = model.predict(features)
select {request.id, shipment_cost}
into source(kafka("127.0.0.1:9092", "cost_predictions"), json());
```
