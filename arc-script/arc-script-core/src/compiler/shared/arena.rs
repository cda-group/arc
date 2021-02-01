#[derive(Debug)]
pub struct Arena<T> {
    pub(crate) store: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            store: Vec::default(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ArenaId(usize);

impl<T> Arena<T> {
    pub fn intern(&mut self, expr: T) -> ArenaId {
        let id = ArenaId(self.store.len());
        self.store.push(expr);
        id
    }

    pub fn resolve(&self, id: ArenaId) -> &T {
        self.store.get(id.0).unwrap()
    }
}
