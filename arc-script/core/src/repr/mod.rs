pub(crate) mod ast;
pub(crate) mod dataflow;
pub(crate) mod info;
pub(crate) mod grammar {
    #[allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/repr/grammar.rs"));
}
