use ast::Type;

use super::blob::blob;
use super::file::file;
use super::matrix::matrix;
use super::rust;
use super::t;
use super::tc;

pub(crate) fn model() -> Type {
    t("Model")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Model", [], [rust("Model")])
        .f("load_model", [], [blob()], model(), [rust("Model::load")])
        .f("predict", ["I", "O"], [model(), matrix(t("I"))], matrix(t("O")), [rust("Model::predict")]);
}
