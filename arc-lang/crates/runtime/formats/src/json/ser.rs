use crate::Encode;

pub struct Writer {}

impl Writer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Encode for Writer {
    type Error = serde_json::Error;

    fn encode<T>(&mut self, input: &T, output: &mut Vec<u8>) -> Result<usize, Self::Error>
    where
        T: serde::Serialize + ?Sized,
    {
        let mut serializer = serde_json::Serializer::new(output);
        input.serialize(&mut serializer)?;
        Ok(1)
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }
}
