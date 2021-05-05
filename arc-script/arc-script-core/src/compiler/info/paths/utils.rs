use crate::compiler::info::paths::Path;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::files::Loc;

impl From<PathId> for Path {
    fn from(id: PathId) -> Self {
        Self::new(id, Loc::Fake)
    }
}
