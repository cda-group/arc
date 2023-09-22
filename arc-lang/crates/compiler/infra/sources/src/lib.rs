use std::fmt::Debug;
use std::fmt::Display;

use ariadne::Cache;
use ariadne::Source;

#[derive(Default)]
pub struct Sources {
    sources: Vec<(String, Source)>,
}

impl Sources {
    pub fn new() -> Self {
        Self { sources: vec![] }
    }
    pub fn add(&mut self, name: impl Into<String>, source: impl Into<String>) -> usize {
        let id = self.sources.len();
        self.sources
            .push((name.into(), Source::from(source.into())));
        id
    }
    pub fn len(&self) -> usize {
        self.sources.len()
    }
}

impl std::fmt::Debug for Sources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sources")
            .field(
                "sources",
                &self.sources.iter().map(|x| &x.0).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Cache<usize> for Sources {
    fn fetch(&mut self, id: &usize) -> Result<&Source, Box<dyn Debug + '_>> {
        self.sources.get(*id).map(|x| &x.1).ok_or_else(|| {
            Box::new(format!("Source with id {} not found in cache", id))
                as Box<dyn std::fmt::Debug + '_>
        })
    }

    fn display<'a>(&self, id: &'a usize) -> Option<Box<dyn Display + 'a>> {
        self.sources
            .get(*id)
            .map(|x| Box::new(x.0.clone()) as Box<dyn Display + 'a>)
    }
}
