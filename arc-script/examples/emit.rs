use arc_script::{compile, opt::*};
fn main() {
    let mut args = std::env::args();
    let _ = args.next();
    let ref source = args.next().unwrap();
    let opt = Opt {
        debug: false,
        mlir: false,
        verbose: false,
        subcmd: SubCmd::Lib,
    };
    let (script, reporter) = arc_script::compile(source, &opt);
    println!("{}", reporter.emit_as_str());
}
