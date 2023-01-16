use dataflow::prelude::*;

use custom_operator::CustomOperator;

fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let kind = args.next();
    let db = Database::local("hello.txt");
    match kind.unwrap().as_str() {
        "worker0" => worker0(db),
        "worker1" => worker1(db),
        "worker2" => worker2(db),
        "worker3" => worker3(db),
        _ => panic!("Unknown arg, expected main[0-3]"),
    }
}

fn worker0(db: Database) {
    Runtime::new()
        .spawn(instance0(db.clone()), 0)
        .spawn(instance0(db.clone()), 1);
}

fn worker1(db: Database) {
    Runtime::new()
        .spawn(instance1(8080, db.clone()), 0)
        .spawn(instance1(8081, db.clone()), 1);
}

fn worker2(db: Database) {
    Runtime::new()
        .spawn(instance2(8082, db.clone()), 0)
        .spawn(instance2(8083, db.clone()), 1);
}

fn worker3(db: Database) {
    Runtime::new()
        .spawn(instance3(db.clone()), 0)
        .spawn(instance3(db.clone()), 1);
}

// kafka-source -> filter -> map -> apply(<user defined operator>) -> group_by
async fn instance0(db: Database) {
    let mut worker = WorkerSource::new();
    let mut source = KafkaSource::<i32, i32>::new("127.0.0.1:9093", "temperature", 0..10);
    let mut sink = ShuffleSink::new(["127.0.0.1:8000", "127.0.0.1:8001"], key_udf).await;

    let (param0, state0) = Filter::new(filter_udf);
    let (param1, state1) = Map::new(map_udf);
    let (param2, state2) = CustomOperator::new(agg_udf);
    let (param3, state3) = Apply::new();

    let param = (param0, param1, param2, param3);
    let mut states = State::new("instance0", db, (state0, state1, state2, state3));

    loop {
        select! {
            event = source.recv() => match event {
                Some((key, mut batch)) => {
                    let state = states.get(key);
                    let mut iter0 = Filter::process(&mut batch, param.0, state.0);
                    let mut iter1 = Map::process(&mut iter0, param.1, state.1);
                    let mut iter2 = CustomOperator::process(param.2, state.2);
                    let mut iter3 = Apply::process(&mut iter1, &mut iter2, param.3, state.3);
                    sink.send(&mut iter3).await;
                    state.3 = iter3.state();
                    state.2 = iter2.state();
                    state.1 = iter1.state();
                    state.0 = iter0.state();
                }
                None => break,
            },
            event = worker.recv() => match event {
                Some(PipelineEvent::Epoch(id)) => {
                    states.persist().await;
                    sink.send_epoch(id).await;
                },
                None => break,
            }
        }
    }
}

// group-by -> filter -> scan -> apply(<user defined operator>) -> group_by
async fn instance1(port: u16, db: Database) {
    let mut source = ShuffleSource::new(port, 2, key_udf).await;
    let mut sink = ShuffleSink::new(["127.0.0.1:8002", "127.0.0.1:8003"], key_udf).await;

    let (param0, state0) = Filter::new(filter_udf);
    let (param1, state1) = Scan::new(agg_udf, 0);
    let (param2, state2) = CustomOperator::new(agg_udf);
    let (param3, state3) = Apply::new();

    let params = (param0, param1, param2, param3);
    let mut states = State::new("instance1", db, (state0, state1, state2, state3));

    while let Some((key, mut batch)) = source.recv().await {
        let state = states.get(key);
        let mut iter0 = Filter::process(&mut batch, params.0, state.0);
        let mut iter1 = Scan::process(&mut iter0, params.1, state.1);
        let mut iter2 = CustomOperator::process(params.2, state.2);
        let mut iter3 = Apply::process(&mut iter1, &mut iter2, params.3, state.3);
        sink.send(&mut iter3).await;
        state.3 = iter3.state();
        state.2 = iter2.state();
        state.1 = iter1.state();
        state.0 = iter0.state();
    }
}

// group-by -> filter -> scan -> apply(<user defined operator>) -> kafka-sink
async fn instance2(port: u16, db: Database) {
    let mut source = ShuffleSource::new(port, 10, key_udf).await;
    let mut sink = KafkaSink::new("127.0.0.1:9093", "output", key_udf);

    let (param0, state0) = Filter::new(filter_udf);
    let (param1, state1) = Scan::new(agg_udf, 0);
    let (param2, state2) = CustomOperator::new(agg_udf);
    let (param3, state3) = Apply::new();

    let params = (param0, param1, param2, param3);
    let mut states = State::new("instance1", db, (state0, state1, state2, state3));

    while let Some((key, mut input)) = source.recv().await {
        let state = states.get(key);
        let mut iter0 = Filter::process(&mut input, params.0, state.0);
        let mut iter1 = Scan::process(&mut iter0, params.1, state.1);
        let mut iter2 = CustomOperator::process(params.2, state.2);
        let mut iter3 = Apply::process(&mut iter1, &mut iter2, params.3, state.3);
        sink.send(&mut iter3).await;
        state.3 = iter3.state();
        state.2 = iter2.state();
        state.1 = iter1.state();
        state.0 = iter0.state();
    }
}

// kafka-source -> filter -+
//                         v
//                       union -> aggregate -> kafka-sink
//                         ^
// kafka-source -> map ----+
async fn instance3(db: Database) {
    let mut source0 = KafkaSource::<i32, _>::new("127.0.0.1:9093", "people", 0..10);
    let mut source1 = KafkaSource::<i32, _>::new("127.0.0.1:9093", "accounts", 0..10);
    let mut sink = KafkaSink::new("127.0.0.1:9093", "output", key_udf);

    let (param0, state0) = Filter::new(filter_udf);
    let (param1, state1) = Union::new();

    let param = (param0, param1);
    let mut states = State::new("instance3", db, (state0, state1));

    loop {
        select! {
            Some((key, mut input)) = source0.recv() => {
                let state = states.get(key);
                let mut iter0 = Filter::process(&mut input, param.0, state.0);
                let mut iter1 = Union::process(&mut iter0, iter::empty(), param.1, state.1);
                sink.send(&mut iter1).await;
                state.1 = iter1.state();
                state.0 = iter0.state();
            },
            Some((key, mut input)) = source1.recv() => {
                let state = states.get(key);
                let mut iter0 = Union::process(&mut input, iter::empty(), param.1, state.1);
                sink.send(&mut iter0).await;
                state.1 = iter0.state();
            },
            else => break
        }
    }
}

// UDFs

fn key_udf(x: i32) -> i32 {
    x % 2
}

fn map_udf(x: i32) -> i32 {
    x + 1
}

fn filter_udf(x: i32) -> bool {
    x % 2 == 0
}

fn agg_udf(x: i32, agg: i32) -> i32 {
    x + agg
}

// UDOs

mod custom_operator {
    use dataflow::prelude::*;

    use serde::Deserialize;
    use serde::Serialize;

    pub struct CustomOperator<I, O> {
        param: Param<I, O>,
        state: State<I, O>,
    }

    impl<I, O> CustomOperator<I, O> {
        pub const fn new(fun: fn(I, I) -> O) -> (Param<I, O>, State<I, O>) {
            (Param { fun }, State::Recv0)
        }
    }

    impl<I, O> CustomOperator<I, O> {
        pub const fn process(param: Param<I, O>, state: State<I, O>) -> Self {
            Self { param, state }
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum State<I, O> {
        Recv0,
        Recv1 { input0: I },
        Send0 { input0: I, input1: I, output0: O },
        Send1 { output1: O },
    }

    #[derive(Copy, Clone)]
    pub struct Param<I, O> {
        fun: fn(I, I) -> O,
    }

    impl<I: Data, O: Data> Iteratee for CustomOperator<I, O> {
        type Item = I;
        fn feed(&mut self, input: Self::Item) {
            match self.state.clone() {
                State::Recv0 => {
                    self.state = State::Recv1 { input0: input };
                }
                State::Recv1 { input0 } => {
                    self.state = State::Send0 {
                        input0: input0.clone(),
                        input1: input.clone(),
                        output0: (self.param.fun)(input0, input),
                    };
                }
                _ => {}
            }
        }
    }

    impl<I: Data, O: Data> Iterator for CustomOperator<I, O> {
        type Item = O;

        fn next(&mut self) -> Option<Self::Item> {
            match self.state.clone() {
                State::Send0 {
                    input0,
                    input1,
                    output0,
                } => {
                    let output1 = (self.param.fun)(input0.clone(), input1.clone());
                    self.state = State::Send1 { output1 };
                    Some(output0)
                }
                State::Send1 { output1 } => {
                    self.state = State::Recv0;
                    Some(output1)
                }
                _ => None,
            }
        }
    }

    impl<I: Data, O: Data> Operator for CustomOperator<I, O> {
        type S = State<I, O>;
        fn state(self) -> Self::S {
            self.state
        }
    }
}
