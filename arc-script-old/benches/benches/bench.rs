//! Benchmarks of compilation time.

use arc_script_compiler::prelude;
use arc_script_compiler::prelude::diags::Sink;
use arc_script_compiler::prelude::modes::{Input, Mode, Output};

use criterion::*;
use include_dir::*;

const SCRIPTS_DIR: Dir = include_dir!("benches/scripts/");

criterion_group!(
    name = benches;
    config = Criterion::default()
               .sample_size(1000)
               .with_plots()
               .warm_up_time(std::time::Duration::new(3, 0));
    targets = end_to_end
);

criterion_main!(benches);

fn end_to_end(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("end_to_end");
    group.sample_size(100);
    let mut sink = Sink::default();
    for (i, script) in SCRIPTS_DIR.files().iter().enumerate() {
        let path = script.path();
        let id = BenchmarkId::new(path.to_str().unwrap(), i);
        let mode = Mode {
            input: Input::File(Some(path.to_path_buf())),
            output: Output::HIR,
            ..Default::default()
        };
        group.bench_with_input(id, &mode, |bench, opt| {
            bench.iter(|| compiler::compile(black_box(opt.clone()), &mut sink))
        });
    }
}
