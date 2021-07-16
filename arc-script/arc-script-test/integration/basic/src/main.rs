#![feature(unboxed_closures)]
#![allow(deprecated)]

use arc_script::arcorn::arctime::prelude::*;

#[allow(clippy::all)]
#[allow(unused_parens)]
#[allow(non_camel_case_types)]
mod script {
    arc_script::include!("src/main.rs");
}

fn main() {
    let executor = Executor::new();

    let pipeline = executor.pipeline();

    let data = (0..100).map(|x| (DateTime::now(), (0, x).into()));

    let stream = pipeline.source(data, Duration::from_millis(100));
    let stream = script::pipe(stream);

    stream.log_sink();

    pipeline.build();

    executor.execute();
}
