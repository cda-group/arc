use crate::traits::DeepClone;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use time::format_description::well_known;

use super::string::String;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
#[repr(C)]
pub struct Time(pub time::OffsetDateTime);

const EU: &[time::format_description::FormatItem<'_>] =
    time::macros::format_description!(version = 2, "[year]-[month]-[day] [hour]:[minute]:[second]");

const US: &[time::format_description::FormatItem<'_>] = time::macros::format_description!(
    version = 2,
    "[month]/[day]/[year] [hour]:[minute]:[second] [period case:upper]"
);

impl Serialize for Time {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let v = time::OffsetDateTime::format(self.0, &well_known::Iso8601::DEFAULT)
            .map_err(serde::ser::Error::custom)?;
        v.serialize(s)
    }
}

impl<'de> Deserialize<'de> for Time {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Time, D::Error> {
        let s: std::string::String = Deserialize::deserialize(d)?;
        match time::PrimitiveDateTime::parse(s.as_ref(), &well_known::Iso8601::DEFAULT)
            .or_else(|_| time::PrimitiveDateTime::parse(s.as_ref(), &well_known::Rfc2822))
            .or_else(|_| time::PrimitiveDateTime::parse(s.as_ref(), &well_known::Rfc3339))
            .or_else(|_| time::PrimitiveDateTime::parse(s.as_ref(), &EU))
            .or_else(|_| time::PrimitiveDateTime::parse(s.as_ref(), &US))
            .map_err(serde::de::Error::custom)
        {
            Ok(v) => Ok(Time(v.assume_utc())),
            Err(e) => s
                .as_str()
                .parse()
                .ok()
                .and_then(|v| time::OffsetDateTime::from_unix_timestamp(v).ok())
                .map(|v| Time(v))
                .ok_or(e),
        }
    }
}

impl DeepClone for Time {
    fn deep_clone(&self) -> Self {
        Time(self.0)
    }
}

impl Time {
    pub fn now() -> Time {
        Time(time::OffsetDateTime::now_utc())
    }

    pub fn from_seconds(seconds: i64) -> Time {
        Time(time::OffsetDateTime::from_unix_timestamp(seconds).unwrap())
    }

    pub fn from_nanoseconds(nanoseconds: i128) -> Time {
        Time(time::OffsetDateTime::from_unix_timestamp_nanos(nanoseconds).unwrap())
    }

    pub fn from_string(text: String, format: String) -> Time {
        let format = time::format_description::parse_owned::<2>(format.as_ref()).unwrap();
        Time(time::OffsetDateTime::parse(text.as_ref(), &format).unwrap())
    }

    pub fn seconds(self) -> i64 {
        self.0.unix_timestamp()
    }

    pub fn nanoseconds(self) -> i128 {
        self.0.unix_timestamp_nanos()
    }

    pub fn year(self) -> i32 {
        self.0.year() as i32
    }

    pub fn to_text(self, format: String) -> String {
        let format = time::format_description::parse_owned::<2>(format.as_ref()).unwrap();
        String::from(self.0.format(&format).unwrap().as_str())
    }
}
