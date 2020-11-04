use arc_script::prelude::*;
fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let source = &args.next().unwrap();
    let opt = Opt::default();
    let script = compiler::compile(source, &opt);
    println!("{}", script.emit_as_str());
}
