use crate::compiler::shared::Map;
use crate::compiler::shared::New;
use crate::compiler::shared::Set;

use syn::Item;

#[derive(Debug, Default, New)]
pub(crate) struct Rust {
    pub(crate) items: Vec<Item>,
}
