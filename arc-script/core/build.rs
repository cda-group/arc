extern crate lalrpop;

fn main() {
    lalrpop::Configuration::new()
        .emit_whitespace(false)
        .use_cargo_dir_conventions()
        .process()
        .unwrap();
}
