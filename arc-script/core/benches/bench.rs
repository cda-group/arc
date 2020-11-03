use {
    arc_script::{compile, opt::*},
    criterion::*,
    include_dir::*,
};

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
    let opt = &Opt::default();
    group.sample_size(100);
    for (i, script) in SCRIPTS_DIR.files().iter().enumerate() {
        let path = &script.path();
        let source = &script.contents_utf8().expect("Failed reading file");
        let id = BenchmarkId::new(path.to_str().unwrap(), i);
        group.bench_with_input(id, source, |bench, source| {
            bench.iter(|| compile(black_box(source), black_box(opt)))
        });
    }
}
