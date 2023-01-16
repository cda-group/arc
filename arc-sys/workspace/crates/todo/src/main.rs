use pipeline::prelude::*;
fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let kind = args.next();
    let db = Database::remote("127.0.0.1:2379");
    match kind.unwrap().as_str() {
        "worker2" => worker2(db),
        "worker0" => worker0(db),
        _ => panic!("Unknown arg, expected instance[N]"),
    }
}
fn worker2(db: Database) {
    Runtime::new()
        .spawn(worker2_shard0(db.clone()), 0usize)
        .spawn(worker2_shard1(db.clone()), 1usize);
}
async fn worker2_shard0(db: Database) {
    let mut node3 = ShuffleSource::new(8000u16, 2usize, key_udf).await;
    let (param0, state0) = Map::new(map_udf);
    let mut node5 = KafkaSink::new("127.0.0.1:9093", "my-topic-2", key_udf);
    let param = (param0,);
    let mut state = State::new("worker2_shard0", db, (state0,));
    while let Some((key, mut node3)) = node3.recv().await {
        let mut state = state.get(key);
        let mut node4 = Map::process(&mut node3, param.0, state.0);
        node5.send(&mut node4).await;
        state.0 = node4.state();
    }
}
async fn worker2_shard1(db: Database) {
    let mut node3 = ShuffleSource::new(8001u16, 2usize, key_udf).await;
    let (param0, state0) = Map::new(map_udf);
    let mut node5 = KafkaSink::new("127.0.0.1:9093", "my-topic-2", key_udf);
    let param = (param0,);
    let mut state = State::new("worker2_shard1", db, (state0,));
    while let Some((key, mut node3)) = node3.recv().await {
        let mut state = state.get(key);
        let mut node4 = Map::process(&mut node3, param.0, state.0);
        node5.send(&mut node4).await;
        state.0 = node4.state();
    }
}
fn worker0(db: Database) {
    Runtime::new()
        .spawn(worker0_shard0(db.clone()), 0usize)
        .spawn(worker0_shard1(db.clone()), 1usize);
}
async fn worker0_shard0(db: Database) {
    let mut node0 = KafkaSource::<i32, i32>::new("127.0.0.1:9093", "my-topic-1", 0..10i32);
    let (param0, state0) = Filter::new(filter_udf);
    let mut node2 = ShuffleSink::new(["127.0.0.1:8000", "127.0.0.1:8001"], key_udf).await;
    let param = (param0,);
    let mut state = State::new("worker0_shard0", db, (state0,));
    while let Some((key, mut node0)) = node0.recv().await {
        let mut state = state.get(key);
        let mut node1 = Filter::process(&mut node0, param.0, state.0);
        node2.send(&mut node1).await;
        state.0 = node1.state();
    }
}
async fn worker0_shard1(db: Database) {
    let mut node0 = KafkaSource::<i32, i32>::new("127.0.0.1:9093", "my-topic-1", 0..10i32);
    let (param0, state0) = Filter::new(filter_udf);
    let mut node2 = ShuffleSink::new(["127.0.0.1:8000", "127.0.0.1:8001"], key_udf).await;
    let param = (param0,);
    let mut state = State::new("worker0_shard1", db, (state0,));
    while let Some((key, mut node0)) = node0.recv().await {
        let mut state = state.get(key);
        let mut node1 = Filter::process(&mut node0, param.0, state.0);
        node2.send(&mut node1).await;
        state.0 = node1.state();
    }
}
fn filter_udf(x: i32) -> bool {
    x > 0
}
fn map_udf(x: i32) -> i32 {
    x + 1
}
fn key_udf(x: i32) -> i32 {
    x % 2
}
