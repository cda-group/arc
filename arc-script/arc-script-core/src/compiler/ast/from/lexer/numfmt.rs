use lexical_core::NumberFormat;

pub(crate) fn compile() -> NumberFormat {
    // Let's use the standard, Rust grammar.
    let mut fmt = NumberFormat::standard().unwrap();
    fmt.set(NumberFormat::NO_SPECIAL, true);
    fmt
}
