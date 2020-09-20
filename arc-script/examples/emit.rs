use arc_script::opt::*;
fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let ref source = args.next().unwrap();
    let opt = Opt {
        debug: false,
        mlir: false,
        verbose: false,
        check: false,
        subcmd: SubCmd::Lib,
    };
    let script = arc_script::compile(source, &opt);
    println!("{}", script.emit_as_str());
}
