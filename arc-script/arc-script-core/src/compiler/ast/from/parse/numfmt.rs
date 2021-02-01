use lexical_core::NumberFormat;

pub(crate) fn compile() -> NumberFormat {
    // Let's use the standard, Rust grammar.
    NumberFormat::standard().unwrap()
}
