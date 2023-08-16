use std::io::Result;
use std::io::Write;

use ast::Type;

use super::duration::duration;
use super::i32::i32;
use super::rust;
use super::t;

pub(crate) fn discretizer() -> Type {
    t("Discretizer")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Discretizer", [], [rust("Discretizer")])
        .f("tumbling", [], [duration()], discretizer(), [rust("Discretizer::tumbling")])
        .f("sliding", [], [duration(), duration()], discretizer(), [rust("Discretizer::sliding")])
        .f("session", [], [duration()], discretizer(), [rust("Discretizer::session")])
        .f("counting", [], [i32()], discretizer(), [rust("Discretizer::counting")])
        .f("moving", [], [i32(), i32()], discretizer(), [rust("Discretizer::moving")]);
}
