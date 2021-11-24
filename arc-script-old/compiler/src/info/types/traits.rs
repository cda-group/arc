//! Built-in traits for different kinds of types.

use crate::hir::ItemKind;
use crate::hir::Path;
use crate::hir::ScalarKind;
use crate::hir::Type;
use crate::hir::TypeId;
use crate::hir::TypeKind;
use crate::hir::HIR;
use crate::info::Info;

use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Shrinkwrap;

use ScalarKind::*;
use TypeKind::*;

#[derive(From, Copy, Clone, Shrinkwrap)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i Info,
    hir: &'i HIR,
}

impl Path {
    /// Returns `true` if a path points to something which can be copied, else `false`.
    /// NOTE: Function pointers and external function pointers are copyable.
    #[rustfmt::skip]
    fn is_copyable<'a>(&self, ctx: impl Into<Context<'a>>) -> bool {
        let ctx = ctx.into();
        match &ctx.hir.resolve(self).kind {
            ItemKind::TypeAlias(item)  => item.t.is_copyable(ctx),
            ItemKind::Enum(item)       => item.variants.iter().all(|v| v.is_copyable(ctx)),
            ItemKind::Fun(item)        => true,
            ItemKind::Task(item)       => false,
            ItemKind::ExternFun(item)  => true,
            ItemKind::ExternType(item) => true,
            ItemKind::Variant(item)    => item.t.is_copyable(ctx),
        }
    }
}

impl TypeId {
    /// Returns `true` if type can be copied, else `false`.
    #[rustfmt::skip]
    pub(crate) fn is_copyable<'a>(self, ctx: impl Into<Context<'a>>) -> bool {
        let ctx = ctx.into();
        match ctx.types.resolve(self) {
            Array(_, _) => false,
            Fun(_, _)   => true,
            Nominal(x)  => false,
            Scalar(Str) => false,
            Scalar(_)   => true,
            Stream(_)   => false,
            Struct(fs)  => fs.values().all(|t| t.is_copyable(ctx)),
            Tuple(ts)   => ts.iter().all(|t| t.is_copyable(ctx)),
            Unknown(_)  => false,
            Err         => true,
        }
    }
    /// Returns `true` if type is an integer, else `false`.
    pub(crate) fn is_int(self, info: &Info) -> bool {
        self.is_sint(info) || self.is_uint(info)
    }
    /// Returns `true` if type is a signed integer, else `false`.
    pub(crate) fn is_sint(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self) {
            matches!(kind, I8 | I16 | I32 | I64)
        } else {
            false
        }
    }
    /// Returns `true` if type is an unsigned integer, else `false`.
    pub(crate) fn is_uint(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self) {
            matches!(kind, U8 | U16 | U32 | U64)
        } else {
            false
        }
    }
    /// Returns `true` if type is a float, else `false`.
    pub(crate) fn is_float(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self) {
            matches!(kind, F32 | F64)
        } else {
            false
        }
    }
    /// Returns `true` if type is a boolean, else `false`.
    pub(crate) fn is_bool(self, info: &Info) -> bool {
        matches!(info.types.resolve(self), Scalar(Bool))
    }
    pub(crate) fn is_unit(self, info: &Info) -> bool {
        matches!(info.types.resolve(self), Scalar(Unit))
    }
    pub(crate) fn is_fun(self, info: &Info) -> bool {
        matches!(info.types.resolve(self), Fun(_, _))
    }
}
