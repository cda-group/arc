use serde::de::DeserializeSeed;
use serde::Deserialize;

use crate::Decode;

pub struct Reader<const N: usize> {
    inner: csv_core::Reader,
    buffer: [u8; N],
}

struct Deserializer<'a, const N: usize> {
    reader: &'a mut Reader<N>,
    input: &'a [u8],
    nread: usize,
    record_end: bool,
    peeked: Option<usize>,
}

impl<const N: usize> Reader<N> {
    #[allow(clippy::new_without_default)]
    pub fn new(sep: char) -> Self {
        Self {
            inner: csv_core::ReaderBuilder::new().delimiter(sep as u8).build(),
            buffer: [0; N],
        }
    }
}
impl<const N: usize> Decode for Reader<N> {
    type Error = Error;
    fn decode<'de, T>(&mut self, input: &'de [u8]) -> Result<T>
    where
        T: Deserialize<'de>,
    {
        let mut deserializer = Deserializer::new(self, input);
        T::deserialize(&mut deserializer)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Buffer overflow.
    Overflow,
    /// Expected an empty field.
    ExpectedEmpty,
    /// Invalid boolean value. Expected either `true` or `false`.
    InvalidBool(String),
    /// Invalid integer.
    InvalidInt(String),
    /// Invalid floating-point number.
    InvalidFloat(lexical_parse_float::Error),
    /// Invalid UTF-8 encoded character.
    InvalidChar(String),
    /// Invalid UTF-8 encoded string.
    InvalidStr(std::str::Utf8Error),
    /// Invalid UTF-8 encoded string.
    InvalidString(std::string::FromUtf8Error),
    /// Error with a custom message had to be discard.
    Custom(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "Buffer overflow."),
            Self::ExpectedEmpty => write!(f, "Expected an empty field."),
            Self::InvalidBool(s) => write!(f, "Invalid bool: {s}"),
            Self::InvalidInt(s) => write!(f, "Invalid integer: {s}"),
            Self::InvalidFloat(e) => write!(f, "Invalid float: {e}"),
            Self::InvalidChar(s) => write!(f, "Invalid character: {s}"),
            Self::InvalidStr(e) => write!(f, "Invalid string: {e}"),
            Self::InvalidString(e) => write!(f, "Invalid string: {e}"),
            Self::Custom(s) => write!(f, "CSV does not match deserializer's expected format: {s}"),
        }
    }
}

impl serde::de::StdError for Error {}

impl serde::de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self::Custom(msg.to_string())
    }
}

impl<'a, const N: usize> Deserializer<'a, N> {
    pub fn new(reader: &'a mut Reader<N>, input: &'a [u8]) -> Self {
        Self {
            reader,
            input,
            nread: 0,
            record_end: false,
            peeked: None,
        }
    }

    /// Read a single field from the CSV input.
    fn advance(&mut self) -> Result<usize> {
        let (result, r, w) = self
            .reader
            .inner
            .read_field(&self.input[self.nread..], &mut self.reader.buffer);
        self.nread += r;
        match result {
            csv_core::ReadFieldResult::InputEmpty => {}
            csv_core::ReadFieldResult::OutputFull => return Err(Error::Overflow),
            csv_core::ReadFieldResult::Field { record_end } => self.record_end = record_end,
            csv_core::ReadFieldResult::End => {}
        }
        Ok(w)
    }

    fn peek_bytes(&mut self) -> Result<&[u8]> {
        let len = match self.peeked {
            Some(len) => len,
            None => {
                let len = self.advance()?;
                self.peeked = Some(len);
                len
            }
        };
        Ok(&self.reader.buffer[..len])
    }

    fn read_bytes(&mut self) -> Result<&[u8]> {
        let len = match self.peeked.take() {
            Some(len) => len,
            None => self.advance()?,
        };
        Ok(&self.reader.buffer[..len])
    }

    fn read_int<T: atoi::FromRadix10SignedChecked>(&mut self) -> Result<T> {
        let bytes = self.read_bytes()?;
        atoi::atoi(bytes)
            .ok_or_else(|| Error::InvalidInt(std::str::from_utf8(bytes).unwrap().to_string()))
    }

    fn read_float<T: lexical_parse_float::FromLexical>(&mut self) -> Result<T> {
        T::from_lexical(self.read_bytes()?)
            .map_err(|e: lexical_parse_float::Error| Error::InvalidFloat(e))
    }

    fn read_bool(&mut self) -> Result<bool> {
        let bytes = self.read_bytes()?;
        match bytes {
            b"true" => Ok(true),
            b"false" => Ok(false),
            _ => Err(Error::InvalidBool(
                std::str::from_utf8(bytes).unwrap().to_string(),
            )),
        }
    }

    fn read_char(&mut self) -> Result<char> {
        let str = self.read_str()?;
        let mut iter = str.chars();
        let c = iter
            .next()
            .ok_or_else(|| Error::InvalidChar(str.to_string()))?;
        if iter.next().is_some() {
            return Err(Error::InvalidChar(str.to_string()));
        } else {
            Ok(c)
        }
    }

    fn read_str(&mut self) -> Result<&str> {
        std::str::from_utf8(self.read_bytes()?)
            .map_err(|e: std::str::Utf8Error| Error::InvalidStr(e))
    }

    fn read_string(&mut self) -> Result<String> {
        std::string::String::from_utf8(self.read_bytes()?.to_vec())
            .map_err(|e| Error::InvalidString(e))
    }
}

impl<'de, 'a, 'b, const N: usize> serde::de::Deserializer<'de> for &'a mut Deserializer<'b, N> {
    type Error = Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        unreachable!("`Deserializer::deserialize_any` is not supported")
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_bool(self.read_bool()?)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_i8(self.read_int()?)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_i16(self.read_int()?)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_i32(self.read_int()?)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_i64(self.read_int()?)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_u8(self.read_int()?)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_u16(self.read_int()?)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_u32(self.read_int()?)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_u64(self.read_int()?)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_f32(self.read_float()?)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_f64(self.read_float()?)
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_char(self.read_char()?)
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_str(self.read_str()?)
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_string(self.read_string()?)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_bytes(self.read_bytes()?)
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_byte_buf(self.read_bytes()?.to_vec())
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        if self.peek_bytes()?.is_empty() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        if !self.read_bytes()?.is_empty() {
            return Err(Error::ExpectedEmpty);
        }
        visitor.visit_unit()
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_seq(self)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_enum(self)
    }

    fn deserialize_identifier<V>(self, _visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        unimplemented!("`Deserializer::deserialize_identifier` is not supported");
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: serde::de::Visitor<'de>,
    {
        let _ = self.read_bytes()?;
        visitor.visit_unit()
    }
}

impl<'de, 'a, 'b, const N: usize> serde::de::VariantAccess<'de> for &'a mut Deserializer<'b, N> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        Ok(())
    }

    fn newtype_variant_seed<U: DeserializeSeed<'de>>(self, _seed: U) -> Result<U::Value> {
        unimplemented!("`VariantAccess::newtype_variant_seed` is not implemented");
    }

    fn tuple_variant<V: serde::de::Visitor<'de>>(
        self,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value> {
        unimplemented!("`VariantAccess::tuple_variant` is not implemented");
    }

    fn struct_variant<V: serde::de::Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value> {
        unimplemented!("`VariantAccess::struct_variant` is not implemented");
    }
}

impl<'de, 'a, 'b, const N: usize> serde::de::EnumAccess<'de> for &'a mut Deserializer<'b, N> {
    type Error = Error;

    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: DeserializeSeed<'de>,
    {
        use serde::de::IntoDeserializer;
        let variant_name = self.read_bytes()?;
        seed.deserialize(variant_name.into_deserializer())
            .map(|v| (v, self))
    }
}

impl<'de, 'a, 'b, const N: usize> serde::de::SeqAccess<'de> for &'a mut Deserializer<'b, N> {
    type Error = Error;

    fn next_element_seed<V>(&mut self, seed: V) -> Result<Option<V::Value>>
    where
        V: DeserializeSeed<'de>,
    {
        if self.record_end {
            Ok(None)
        } else {
            seed.deserialize(&mut **self).map(Some)
        }
    }
}
