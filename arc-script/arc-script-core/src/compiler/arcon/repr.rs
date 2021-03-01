use arc_script_core_shared::New;

#[derive(Debug, New)]
pub(crate) struct Arcon {
    pub(crate) file: syn::File,
}
