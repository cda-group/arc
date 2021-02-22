use arcon::prelude::ArconTime;
use arcon::prelude::Pipeline;

#[arc_script::compile("script.arc")]
mod script {}

fn main() {
    let pipeline = Pipeline::default();

    let data = vec![1, 2, 3];

    let stream0 = pipeline.collection(data, |conf| {
        conf.set_arcon_time(ArconTime::Process);
    });

    let stream1 = script::pipe(stream0);

    let mut pipeline = stream1.to_console().build();

    pipeline.start();
    pipeline.await_termination();
}
