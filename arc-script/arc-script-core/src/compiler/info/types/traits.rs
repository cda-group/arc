//! Built-in traits for different kinds of types.

use crate::compiler::hir::ScalarKind;
use crate::compiler::hir::Type;
use crate::compiler::hir::TypeId;
use crate::compiler::hir::TypeKind;
use crate::compiler::info::Info;

use ScalarKind::*;
use TypeKind::*;

impl TypeId {
    /// Returns `true` if type can be copied, else `false`.
    #[rustfmt::skip]
    pub fn is_copyable(self, info: &Info) -> bool {
        match info.types.resolve(self).kind {
            Array(_, _) => false,
            Fun(_, _)   => false,
            Map(_, _)   => false,
            Nominal(_)  => false,
            Optional(t) => t.is_copyable(info),
            Scalar(_)   => true,
            Set(_)      => false,
            Stream(_)   => false,
            Struct(fs)  => fs.values().all(|t| t.is_copyable(info)),
            Tuple(ts)   => ts.iter().all(|t| t.is_copyable(info)),
            Unknown     => false,
            Vector(_)   => false,
            Boxed(_)    => false,
            Err         => true,
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
        if let Scalar(Bool) = info.types.resolve(self).kind {
            true
        } else {
            false
        }
    }
}
