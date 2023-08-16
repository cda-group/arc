use ast::Type;

use super::duration::duration;
use super::function::fun;
use super::rust;
use super::t;
use super::tc;

pub(crate) fn time_source(t: Type) -> Type {
    tc("TimeSource", [t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("TimeSource", ["T"], [rust("TimeSource")])
        .f("ingestion", ["T"], [duration()], time_source(t("T")), [rust("TimeSource::ingestion")])
        .f("event", ["T"], [duration(), duration(), fun([t("T")], duration())], time_source(t("T")), [rust("TimeSource::event")]);
}
