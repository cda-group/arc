//! Built-in traits for different kinds of types.

use crate::compiler::hir::ItemKind;
use crate::compiler::hir::Path;
use crate::compiler::hir::ScalarKind;
use crate::compiler::hir::Type;
use crate::compiler::hir::TypeId;
use crate::compiler::hir::TypeKind;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use arc_script_core_shared::From;

use ScalarKind::*;
use TypeKind::*;

#[derive(From, Copy, Clone)]
pub(crate) struct Context<'i> {
    info: &'i Info,
    hir: &'i HIR,
}

impl Path {
    /// Returns `true` if a path points to something which can be copied, else `false`.
    /// NOTE: Function pointers and external function pointers are copyable.
    #[rustfmt::skip]
    fn is_copyable<'a>(&self, ctx: impl Into<Context<'a>>) -> bool {
        let ctx = ctx.into();
        ctx.hir.defs.get(self).map(|item| {
            match &item.kind {
                ItemKind::Alias(item)   => item.tv.is_copyable(ctx),
                ItemKind::Enum(item)    => item.variants.iter().all(|v| v.is_copyable(ctx)),
                ItemKind::Fun(item)     => true,
                ItemKind::Task(item)    => false,
                ItemKind::Extern(item)  => true,
                ItemKind::Variant(item) => item.tv.is_copyable(ctx),
            }
        }).unwrap_or(true)
    }
}

impl TypeId {
    /// Returns `true` if type can be copied, else `false`.
    #[rustfmt::skip]
    pub(crate) fn is_copyable<'a>(self, ctx: impl Into<Context<'a>>) -> bool {
        let ctx = ctx.into();
        match ctx.info.types.resolve(self).kind {
            Array(_, _)  => false,
            Fun(_, _)    => true,
            Map(_, _)    => false,
            Nominal(x)   => false,
            Optional(t)  => t.is_copyable(ctx),
            Scalar(Str)  => false,
            Scalar(_)    => true,
            Set(_)       => false,
            Stream(_)    => false,
            Struct(fs)   => fs.values().all(|t| t.is_copyable(ctx)),
            Tuple(ts)    => ts.iter().all(|t| t.is_copyable(ctx)),
            Unknown      => false,
            Vector(_)    => false,
            Boxed(_)     => false,
            Err          => true,
        }
    }
    /// Returns `true` if type is an integer, else `false`.
    pub(crate) fn is_int(self, info: &Info) -> bool {
        self.is_sint(info) || self.is_uint(info)
    }
    /// Returns `true` if type is a signed integer, else `false`.
    pub(crate) fn is_sint(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self).kind {
            matches!(kind, I8 | I16 | I32 | I64)
        } else {
            false
        }
    }
    /// Returns `true` if type is an unsigned integer, else `false`.
    pub(crate) fn is_uint(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self).kind {
            matches!(kind, U8 | U16 | U32 | U64)
        } else {
            false
        }
    }
    /// Returns `true` if type is a float, else `false`.
    pub(crate) fn is_float(self, info: &Info) -> bool {
        if let Scalar(kind) = info.types.resolve(self).kind {
            matches!(kind, F16 | Bf16 | F32 | F64)
        } else {
            false
        }
    }
    /// Returns `true` if type is a boolean, else `false`.
    pub(crate) fn is_bool(self, info: &Info) -> bool {
        matches!(info.types.resolve(self).kind, Scalar(Bool))
    }
}
