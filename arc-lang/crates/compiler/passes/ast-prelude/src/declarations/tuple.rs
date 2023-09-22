use ast::Type;
use ast::TypeKind::TTuple;
use im_rc::Vector;
use info::Info;

pub(crate) fn tuple<const N: usize>(ts: [Type; N]) -> Type {
    TTuple(ts.into_iter().collect()).with(Info::Builtin)
}

pub(crate) fn declare(builder: &mut super::Builder) {}
