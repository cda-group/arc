#[derive(Debug)]
pub struct Arena<Id, Kind> {
    pub(crate) store: Vec<Kind>,
    marker: std::marker::PhantomData<Id>,
}

impl<Id, Kind> Default for Arena<Id, Kind> {
    fn default() -> Self {
        Self {
            store: Vec::default(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<Id: From<usize> + Into<usize> + Copy, Kind> Arena<Id, Kind> {
    /// Interns a `Kind` and returns an `Id` to it.
    pub fn intern(&mut self, kind: Kind) -> Id {
        let id = Id::from(self.store.len());
        self.store.push(kind);
        id
    }

    /// Resolves an `Id` into its associated `Kind`.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn resolve(&self, id: impl Into<Id>) -> &Kind {
        let id: Id = id.into();
        self.store.get(id.into()).unwrap()
    }
}
