use super::Info;
use crate::info::types::Type;
use crate::info::types::TypeId;
use lasso::Key;

impl std::fmt::Debug for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Info {{")?;

        writeln!(f, "  Names: [")?;
        self.names
            .store
            .iter()
            .try_for_each(|(id, name)| writeln!(f, "    Name-{:<4?} => {},", id.into_usize(), name))?;
        writeln!(f, "  ],")?;

        writeln!(f, "  Paths: [")?;
        self.paths
            .kind_to_id
            .values()
            .try_for_each(|id| writeln!(f, "    {},", id.debug(self)))?;
        writeln!(f, "  ],")?;

        let ts = self
            .types
            .store
            .borrow()
            .vars_since_snapshot(&self.types.snapshot);
        writeln!(f, "  Types: [")?;
        (ts.start.0..ts.end.0)
            .map(TypeId)
            .map(Type::new)
            .try_for_each(|t| writeln!(f, "    Type-{:<4} => {:?},", t.id.0, self.types.resolve(t)))?;
        writeln!(f, "  ]")?;

        writeln!(f, "}}")?;
        Ok(())
    }
}
