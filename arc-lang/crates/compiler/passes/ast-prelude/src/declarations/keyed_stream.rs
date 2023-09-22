use crate::declarations::vec::vec;
use ast::Type;

use super::aggregator::aggregator;
use super::bool::bool;
use super::dataflow::dataflow;
use super::discretizer::discretizer;
use super::encoding::encoding;
use super::function::fun;
use super::keyed_stream;
use super::reader::reader;
use super::rust;
use super::stream::stream;
use super::t;
use super::tc;
use super::time_source::time_source;
use super::writer::writer;

pub(crate) fn keyed_stream(k: Type, t: Type) -> Type {
    tc("KeyedStream", [k, t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("KeyedStream", ["K", "T"], [rust("KeyedStream")])
        .f("keyed_source", ["K", "T"], [reader(), encoding(), time_source(t("T"))], keyed_stream(t("K"), t("T")), [rust("KeyedStream::source")])
        .f("keyed_sink", ["K", "T"], [keyed_stream(t("K"), t("T")), writer(), encoding()], dataflow(), [rust("KeyedStream::sink")])
        .f("keyed_map", ["K", "A", "B"], [keyed_stream(t("K"), t("A")), fun([t("A")], t("B"))], keyed_stream(t("K"), t("B")), [rust("KeyedStream::map")])
        .f("keyed_filter", ["K", "T"], [keyed_stream(t("K"), t("T")), fun([t("T")], bool())], keyed_stream(t("K"), t("T")), [rust("KeyedStream::filter")])
        .f("keyed_flatmap", ["K", "A", "B"], [keyed_stream(t("K"), vec(t("A"))), fun([t("A")], vec(t("B")))], keyed_stream(t("K"), t("B")), [rust("KeyedStream::flatmap")])
        .f("keyed_flatten", ["K", "T"], [keyed_stream(t("K"), vec(t("A")))], keyed_stream(t("K"), t("B")), [rust("KeyedStream::flatten")])
        .f("keyed_window", ["K", "I", "P", "O"], [keyed_stream(t("K"), t("I")), discretizer(), aggregator(t("I"), t("P"), t("O"))], keyed_stream(t("K"), t("O")), [rust("KeyedStream::window")])
        .f("keyed_keyby", ["K0", "K1", "T"], [stream(t("T")), fun([t("T")], t("K0"))], keyed_stream(t("K1"), t("T")), [rust("KeyedStream::keyby")])
        .f("unkey", ["K", "T"], [keyed_stream(t("K"), t("T"))], stream(t("T")), [rust("KeyedStream::unkey")]);
}
