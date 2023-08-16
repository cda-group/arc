use ast::Type;
use ast::TypeKind::TFun;
use info::Info;

pub(crate) fn fun<const N: usize>(args: [Type; N], ret: Type) -> Type {
    TFun(args.into_iter().collect(), ret).with(Info::Builtin)
}

pub(crate) fn declare(builder: &mut super::Builder) {}
