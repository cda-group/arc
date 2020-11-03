use arc_script::opt::*;
fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let source = &args.next().unwrap();
    let opt = Opt::default();
    let script = arc_script::compile(source, &opt);
    println!("{}", script.emit_as_str());
}
