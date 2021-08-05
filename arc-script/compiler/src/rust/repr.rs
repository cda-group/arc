use arc_script_compiler_shared::New;

#[derive(Debug, New)]
pub(crate) struct Rust {
    pub(crate) file: syn::File,
}
