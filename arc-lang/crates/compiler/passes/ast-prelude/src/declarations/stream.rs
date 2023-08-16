use ast::Type;

use super::aggregator::aggregator;
use super::bool::bool;
use super::dataflow::dataflow;
use super::encoding::encoding;
use super::function::fun;
use super::keyed_stream::keyed_stream;
use super::reader::reader;
use super::rust;
use super::t;
use super::tc;
use super::time_source::time_source;
use super::vec::vec;
use super::writer::writer;

pub(crate) fn stream(t: Type) -> Type {
    tc("Stream", [t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Stream", ["T"], [rust("Stream")])
        .f("source", ["T"], [reader(), encoding(), time_source(t("T"))], stream(t("T")), [rust("Stream::source")])
        .f("sink", ["T"], [stream(t("T")), writer(), encoding()], dataflow(), [rust("Stream::sink")])
        .f("map", ["A", "B"], [stream(t("A")), fun([t("A")], t("B"))], stream(t("B")), [rust("Stream::map")])
        .f("filter", ["T"], [stream(t("T")), fun([t("T")], bool())], stream(t("T")), [rust("Stream::filter")])
        .f("flatmap", ["A", "B"], [stream(vec(t("A"))), fun([t("A")], vec(t("B")))], stream(t("B")), [rust("Stream::flatmap")])
        .f("flatten", ["T"], [stream(vec(t("A")))], stream(t("B")), [rust("Stream::flatten")])
        .f("window", ["I", "P", "O"], [stream(t("I")), t("Discretizer"), aggregator(t("I"), t("P"), t("O"))], stream(t("O")), [rust("Stream::window")])
        .f("keyby", ["K", "T"], [stream(t("T")), fun([t("T")], t("K"))], keyed_stream(t("K"), t("T")), [rust("Stream::keyby")]);
}
