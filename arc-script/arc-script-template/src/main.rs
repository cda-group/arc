use arcon::prelude::ArconTime;
use arcon::prelude::Pipeline;

#[arc_script::compile("script.arc")]
mod script {}

fn main() {
    let pipeline = Pipeline::default();

    let data = vec![1, 2, 3];

    let stream = pipeline.collection(data, |conf| {
        conf.set_arcon_time(ArconTime::Process);
    });

    let stream = script::pipe(stream);

    let mut pipeline = stream.to_console().build();

    pipeline.start();
    pipeline.await_termination();
}

