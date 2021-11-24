use crate::info::paths::Path;
use crate::info::paths::PathId;
use crate::info::files::Loc;

impl From<PathId> for Path {
    fn from(id: PathId) -> Self {
        Self::new(id, Loc::Fake)
    }
}
