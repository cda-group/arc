use crate::val::Value;

#[derive(Debug, Clone)]
pub struct Fun {
    name: String,
    args: Vec<(String, Value)>,
}

impl Fun {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Vec::new(),
        }
    }
    pub fn arg(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.args.push((name.into(), value.into()));
        self
    }
}
