use crate::mlir;

impl mlir::MLIR {
    pub(crate) fn intern(&mut self, path: impl Into<mlir::PathId>, item: mlir::Item) {
        self.defs.insert(path.into(), item);
    }

    pub(crate) fn resolve(&self, path: impl Into<mlir::PathId>) -> &mlir::Item {
        self.defs.get(&path.into()).unwrap()
    }
}
