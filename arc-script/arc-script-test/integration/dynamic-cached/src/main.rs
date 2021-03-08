use arc_script::arcorn::operators::*;
use arcon::prelude::ArconTime;
use arcon::prelude::Pipeline;

mod script {
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

    let stream1 = script::pipe(stream0);

    let mut pipeline = stream1.to_console().build();

    pipeline.start();
    pipeline.await_termination();
}

impl<B: arcon::Backend, C: arcon::prelude::ComponentDefinition>
    script::Identity<'_, '_, '_, '_, B, C>
{
    fn rust_method(&mut self, x: i32) -> i32 {
        self.Identity_arc_method(x);
        x + 5
    }
}
