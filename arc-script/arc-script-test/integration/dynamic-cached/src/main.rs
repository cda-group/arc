use arc_script::arcorn::operators::*;
use arcon::prelude::ArconTime;
use arcon::prelude::Pipeline;

mod test {
    arc_script::include!("src/main.rs");
}

fn main() {
    let pipeline = Pipeline::default();

    let data = vec![1, 2, 3];
    let stream0 = pipeline
        .collection(data, |conf| {
            conf.set_arcon_time(ArconTime::Process);
        })
        .convert();

    let stream1 = test::pipe(stream0);

    let mut pipeline = stream1.to_console().build();

    pipeline.start();
    pipeline.await_termination();
}
