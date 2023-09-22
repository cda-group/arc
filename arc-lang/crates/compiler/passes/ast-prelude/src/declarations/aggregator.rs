use ast::Type;

use super::function::fun;
use super::rust;
use super::t;
use super::tc;
use super::tuple::tuple;

pub(crate) fn aggregator(i: Type, p: Type, o: Type) -> Type {
    tc("Aggregator", [i, p, o])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Aggregator", ["I", "P", "O"], [rust("Aggregator")])
        .f("aggregator", ["I", "P", "O"], [fun([t("I")], t("P")), fun([t("P"), t("P")], t("P")), fun([t("P")], t("O"))], aggregator(t("I"), t("P"), t("O")), [rust("Aggregator::aggregator")])
        // .f(
        //     "compose",
        //     ["I0", "P0", "O0", "I1", "P1", "O1"],
        //     [aggregator(t("I0"), t("P0"), t("O0")), aggregator(t("I1"), t("P1"), t("O1"))],
        //     aggregator(tuple([t("I0"), t("I1")]), tuple([t("P0"), t("P1")]), tuple([t("O0"), t("O1")])),
        //     [rust("Aggregator::compose")],
        // )
        ;
}
