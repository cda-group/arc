use arc_script_core_shared::New;

#[derive(Debug, New)]
pub(crate) struct Rust {
    pub(crate) file: syn::File,
}
