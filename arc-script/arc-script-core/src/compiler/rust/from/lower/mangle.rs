use crate::compiler::hir;
use crate::compiler::shared::VecMap;
use crate::compiler::shared::{New, Set};

use super::Context;

trait Mangle {
    fn mangled_ident(&self, ctx: &mut Context) -> syn::Ident {
        let tmp = std::mem::take(&mut ctx.buf);
        self.mangle(ctx);
        let name = std::mem::take(&mut ctx.buf);
        let span = proc_macro2::Span::call_site();
        ctx.buf = tmp;
        syn::Ident::new(&name, span)
    }
    fn mangle(&self, ctx: &mut Context) -> syn::Ident;
}

/// Mangles an anonymous-struct into a nominal-struct, e.g.
///
/// {a_i32_b: i32}   // Struct7a_i32_b3i32End
/// {a: i32, b: i32} // Struct1a3i321b3i32End
///
// pub(crate) fn mangle_expr_struct(
//     fields: &VecMap<hir::Name, hir::Expr>,
//     ctx: &mut Context,
// ) -> syn::Ident {
//     fields.iter().map(|(x, e)| (x, &e.tv)), ctx)
// }

impl hir::TypeId {
    fn mangle(&self, ctx: &mut Context) {
        match ctx.info.types.resolve(*self).kind {
            hir::TypeKind::Array(t, s) => {
                ctx.buf.push_str("Array_");
                t.mangle(ctx);
                ctx.buf.push_str("_End");
                todo!();
            }
            hir::TypeKind::Fun(ts, t) => {
                ctx.buf.push_str("Fun_");
                ts.iter().for_each(|t| t.mangle(ctx));
                ctx.buf.push_str("_End");
                t.mangle(ctx);
            }
            hir::TypeKind::Map(t0, t1) => todo!(),
            hir::TypeKind::Nominal(x) => todo!(),
            hir::TypeKind::Optional(t) => {
                ctx.buf.push_str("Optional_");
                t.mangle(ctx);
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Scalar(s) => {
                let s = s.as_str(ctx);
                ctx.buf.push_str(s)
            }
            hir::TypeKind::Set(t) => {
                ctx.buf.push_str("Set_");
                t.mangle(ctx);
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Stream(t) => {
                ctx.buf.push_str("Stream_");
                t.mangle(ctx);
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Struct(fts) => {
                let buf = std::mem::take(&mut ctx.buf);
                ctx.buf.push_str("Struct_");
                let mut fts = fts.into_iter().collect::<Vec<_>>();
                fts.sort_by_key(|(x, _)| x.id);
                fts.iter().for_each(|(f, t)| {
                    f.mangle(ctx);
                    ctx.buf.push_str("_");
                    t.mangle(ctx);
                });
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Task(ts0, ts1) => {}
            hir::TypeKind::Tuple(t) => {}
            hir::TypeKind::Unknown => {}
            hir::TypeKind::Vector(t) => {}
            hir::TypeKind::Err => {}
        }
    }
}
// {a:i32, {b:i32, c:i32}}
// Struct_a_i32_Struct_b_i32_c_i32_End_End

impl hir::Name {
    fn mangle(&self, ctx: &mut Context) {
        let name = ctx.info.names.resolve(self.id);
        ctx.buf.push_str(name);
    }
}

impl hir::ScalarKind {
    #[rustfmt::skip]
    fn as_str(&self, ctx: &mut Context) -> &'static str {
        match self {
            hir::ScalarKind::Bool => "bool",
            hir::ScalarKind::Char => "char",
            hir::ScalarKind::Bf16 => "bf16",
            hir::ScalarKind::F16  => "f16",
            hir::ScalarKind::F32  => "f32",
            hir::ScalarKind::F64  => "f64",
            hir::ScalarKind::I8   => "i8",
            hir::ScalarKind::I16  => "i16",
            hir::ScalarKind::I32  => "i32",
            hir::ScalarKind::I64  => "i64",
            hir::ScalarKind::U8   => "u8",
            hir::ScalarKind::U16  => "u16",
            hir::ScalarKind::U32  => "u32",
            hir::ScalarKind::U64  => "u64",
            hir::ScalarKind::Null => todo!(),
            hir::ScalarKind::Str  => todo!(),
            hir::ScalarKind::Unit => "()",
            hir::ScalarKind::Bot  => todo!(),
        }
    }
}
