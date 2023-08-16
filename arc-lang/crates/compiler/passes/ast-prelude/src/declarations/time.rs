use super::rust;
use super::i64::i64;
use super::i128::i128;
use super::string::string;
use super::i32::i32;
use super::Type;
use super::t;

pub(crate) fn time() -> Type {
    t("Time")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Time", [], [rust("Time")])
        .f("now", [], [], time(), [rust("Time::now")])
        .f("from_seconds", [], [i64()], time(), [rust("Time::from_seconds")])
        .f("from_nanoseconds", [], [i128()], time(), [rust("Time::from_nanoseconds")])
        .f("seconds", [], [time()], i64(), [rust("Time::seconds")])
        .f("nanoseconds", [], [time()], i128(), [rust("Time::nanoseconds")])
        .f("year", [], [time()], i32(), [rust("Time::year")])
        .f("from_string", [], [string(), string()], time(), [rust("Time::from_string")])
        .f("into_string", [], [time(), string()], string(), [rust("Time::to_string")]);
}
