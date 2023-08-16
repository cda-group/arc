type Name = String;

#[derive(Clone, Debug)]
pub struct NameGen {
    base: String,
    next: usize,
}

impl NameGen {
    pub fn new(base: impl Into<String>) -> Self {
        NameGen {
            base: base.into(),
            next: 0,
        }
    }
    pub fn fresh(&mut self) -> Name {
        let uid = self.next;
        self.next += 1;
        Name::from(format!("{}{}", self.base, uid))
    }
}
