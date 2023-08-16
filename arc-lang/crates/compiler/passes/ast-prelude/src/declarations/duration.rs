use std::io::Result;
use std::io::Write;

use ast::Type;

use super::i32::i32;
use super::rust;
use super::t;

pub(crate) fn duration() -> Type {
    t("Duration")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Duration", [], [rust("Duration")])
        .f("__s", [], [i32()], duration(), [rust("Duration::seconds")])
        .f("__ms", [], [i32()], duration(), [rust("Duration::milliseconds")])
        .f("__us", [], [i32()], duration(), [rust("Duration::microseconds")])
        .f("__ns", [], [i32()], duration(), [rust("Duration::nanoseconds")])
        .f("from_seconds", [], [i32()], duration(), [rust("Duration::from_seconds")])
        .f("from_milliseconds", [], [i32()], duration(), [rust("Duration::from_milliseconds")])
        .f("from_microseconds", [], [i32()], duration(), [rust("Duration::from_microseconds")])
        .f("from_nanoseconds", [], [i32()], duration(), [rust("Duration::from_nanoseconds")]);
}
