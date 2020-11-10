use super::Info;
use crate::compiler::info::types::TypeId;

impl std::fmt::Debug for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Info {{")?;

        writeln!(f, "  Names: [")?;
        self.names
            .store
            .strings()
            .try_for_each(|name| writeln!(f, r#"    "{}","#, name))?;
        writeln!(f, "  ],")?;

        writeln!(f, "  Paths: [")?;
        self.paths.store.values().try_for_each(|path| {
            writeln!(f, r#"    "{}","#, self.resolve_to_names(*path).join("::"))
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
            .try_for_each(|tv| writeln!(f, "    Type: {:?},", self.types.resolve(tv)))?;
        writeln!(f, "  ]")?;

        writeln!(f, "}}")?;
        Ok(())
    }
}
