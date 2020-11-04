use crate::prelude::*;

pub fn diagnose(source: &str, opt: &Opt) {
    let script = compile(source, opt);
    if script.info.errors.is_empty() && !opt.debug {
        if opt.mlir {
            println!("{}", script.mlir());
        } else {
            println!("{}", script);
        };
    } else {
        script.emit_to_stdout()
    }
}

pub fn compile<'i>(source: &'i str, opt: &'i Opt) -> Script<'i> {
    let mut script = Script::parse(source, opt);

    if opt.debug {
        println!("=== Opt");
        println!("{:?}", opt);
        println!("=== Parsed");
        println!("{}", script);
    }

    // script.body.download();

    script.infer();

    if opt.debug {
        println!("=== Typed");
        println!("{}", script);
    }

    script = script.into_ssa();
    script.prune();

    if opt.debug {
        if opt.debug {
            println!("=== Canonicalized");
            println!("{}", script);
        }

        if script.info.errors.is_empty() {
            println!("=== MLIR");
            println!("{}", script.mlir());
        }
    }

    if !opt.check {
        let dataflow = script.eval();
        println!("{}", dataflow.pretty());
    }

    script
}
