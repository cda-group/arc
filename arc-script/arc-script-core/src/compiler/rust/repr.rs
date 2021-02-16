use arc_script_core_shared::New;

use syn::Item;

#[derive(Debug, Default, New)]
pub(crate) struct Rust {
    pub(crate) items: Vec<Item>,
}
