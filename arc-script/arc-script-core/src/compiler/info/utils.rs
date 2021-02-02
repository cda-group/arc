use super::Info;
use crate::compiler::info::types::TypeId;

impl std::fmt::Debug for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Info {{")?;

        writeln!(f, "  Names: [")?;
        self.names
            .store
            .iter()
            .try_for_each(|(id, name)| writeln!(f, r#"    {:?}: "{}","#, id, name))?;
        writeln!(f, "  ],")?;

        writeln!(f, "  Paths: [")?;
        self.paths.store.values().try_for_each(|id| {
            writeln!(f, r#"    {:?}: "{}","#, id, self.resolve_to_names(*id).join("::"))
        })?;
        writeln!(f, "  ],")?;

        let tvs = self
            .types
            .store
            .borrow()
            .vars_since_snapshot(&self.types.snapshot);
        writeln!(f, "  Types: [")?;
        (tvs.start.0..tvs.end.0)
            .map(TypeId)
            .try_for_each(|tv| writeln!(f, "    {:?}: {:?},", tv, self.types.resolve(tv)))?;
        writeln!(f, "  ]")?;

        writeln!(f, "}}")?;
        Ok(())
    }
}
