use crate::compiler::hir;

use super::Context;

trait Mangle {
    fn mangled_ident(&self, ctx: &mut Context<'_>) -> syn::Ident {
        let tmp = std::mem::take(&mut ctx.buf);
        self.mangle(ctx);
        let name = std::mem::take(&mut ctx.buf);
        let span = proc_macro2::Span::call_site();
        ctx.buf = tmp;
        syn::Ident::new(&name, span)
    }
    fn mangle(&self, ctx: &mut Context<'_>) -> syn::Ident;
}

/// Mangles an anonymous-struct into a nominal-struct, e.g.
///
/// `{a_i32_b: i32}   => Struct7a_i32_b3i32End`
/// `{a: i32, b: i32} => Struct1a3i321b3i32End`
///
// pub(crate) fn mangle_expr_struct(
//     fields: &VecMap<hir::Name, hir::Expr>,
//     ctx: &mut Context,
// ) -> syn::Ident {
//     fields.iter().map(|(x, e)| (x, &e.tv)), ctx)
// }

impl hir::TypeId {
    fn mangle(self, ctx: &mut Context<'_>) {
        match ctx.info.types.resolve(self).kind {
            hir::TypeKind::Boxed(t) => {
                ctx.buf.push_str("Box_");
                t.mangle(ctx);
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Array(t, _s) => {
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
            hir::TypeKind::Map(_t0, _t1) => todo!(),
            hir::TypeKind::Nominal(_x) => todo!(),
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
                let _buf = std::mem::take(&mut ctx.buf);
                ctx.buf.push_str("Struct_");
                let mut fts = fts.into_iter().collect::<Vec<_>>();
                fts.sort_by_key(|(x, _)| x.id);
                fts.iter().for_each(|(f, t)| {
                    f.mangle(ctx);
                    ctx.buf.push('_');
                    t.mangle(ctx);
                });
                ctx.buf.push_str("_End");
            }
            hir::TypeKind::Tuple(_t) => todo!(),
            hir::TypeKind::Unknown => {}
            hir::TypeKind::Vector(_t) => todo!(),
            hir::TypeKind::By(_t0, _1) => todo!(),
            hir::TypeKind::Err => {}
        }
    }
}

impl hir::Name {
    fn mangle(&self, ctx: &mut Context<'_>) {
        let name = ctx.info.names.resolve(self.id);
        ctx.buf.push_str(name);
    }
}

impl hir::ScalarKind {
    #[rustfmt::skip]
    fn as_str(self, _ctx: &mut Context<'_>) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Char => "char",
            Self::Bf16 => "bf16",
            Self::F16  => "f16",
            Self::F32  => "f32",
            Self::F64  => "f64",
            Self::I8   => "i8",
            Self::I16  => "i16",
            Self::I32  => "i32",
            Self::I64  => "i64",
            Self::U8   => "u8",
            Self::U16  => "u16",
            Self::U32  => "u32",
            Self::U64  => "u64",
            Self::Null => todo!(),
            Self::Str  => todo!(),
            Self::Unit => "()",
            Self::Bot  => todo!(),
        }
    }
}
