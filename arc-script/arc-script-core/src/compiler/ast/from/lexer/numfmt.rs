use lexical_core::NumberFormat;

/// Compiles a `NumberFormat` for `lexical_core`. Currently uses the standard Rust grammar.
pub(crate) fn compile() -> NumberFormat {
    let mut fmt = NumberFormat::standard().unwrap();
    fmt.set(NumberFormat::NO_SPECIAL, true);
    fmt
}
