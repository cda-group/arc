use crate::compiler::hir;
use crate::compiler::hir::Name;
use crate::compiler::hir::TypeId;
use crate::compiler::rust::from::lower::Context;
use arc_script_core_shared::get;
use arc_script_core_shared::VecMap;

/// Mangles an anonymous-struct into a nominal-struct, e.g.
///
/// `{a_i32_b: i32}   => Struct7a_i32_b3i32`
/// `{a: i32, b: i32} => Struct1a3i321b3i32`
///
pub(crate) fn mangle(tv: TypeId, ctx: &mut Context<'_>) -> syn::Ident {
    let mut name = String::new();
    let fts = get!(ctx.info.types.resolve(tv).kind, hir::TypeKind::Struct(fs));
    mangle_fields(tv, &fts, ctx, &mut name);
    let span = proc_macro2::Span::call_site();
    syn::Ident::new(&name, span)
}

fn mangle_fields(tv: TypeId, fts: &VecMap<Name, TypeId>, ctx: &mut Context<'_>, name: &mut String) {
    let root = ctx.info.types.root(tv);
    let ident = if let Some(ident) = ctx.mangled_names.get(&root) {
        name.push_str(ident);
    } else {
        let mut inner = String::new();
        // Mangle the struct
        inner.push_str("Struct");
        let mut fts = fts.into_iter().collect::<Vec<_>>();
        fts.sort_by_key(|(x, _)| x.id);
        fts.iter().for_each(|(x, t)| {
            let len = ctx.info.names.resolve(x.id).len();
            inner.push_str(&format!("{}", len));
            x.mangle(ctx, &mut inner);
            t.mangle(ctx, &mut inner);
        });
        inner.push_str("End");
        ctx.mangled_names.insert(root, inner.clone());
        name.push_str(&inner);
    };
}

impl hir::TypeId {
    fn mangle(self, ctx: &mut Context<'_>, name: &mut String) {
        match ctx.info.types.resolve(self).kind {
            hir::TypeKind::Boxed(t) => {
                name.push_str("Box");
                t.mangle(ctx, name);
            }
            hir::TypeKind::Array(t, _s) => {
                name.push_str("Array");
                t.mangle(ctx, name);
                todo!();
            }
            hir::TypeKind::Fun(ts, t) => {
                name.push_str("Fun");
                ts.iter().for_each(|t| t.mangle(ctx, name));
                t.mangle(ctx, name);
            }
            hir::TypeKind::Map(t0, t1) => {
                name.push_str("Map");
            }
            hir::TypeKind::Nominal(x) => {
                x.mangle(ctx, name);
            }
            hir::TypeKind::Optional(t) => {
                name.push_str("Optional");
                t.mangle(ctx, name);
            }
            hir::TypeKind::Scalar(s) => {
                s.mangle(ctx, name);
            }
            hir::TypeKind::Set(t) => {
                name.push_str("Set");
                t.mangle(ctx, name);
            }
            hir::TypeKind::Stream(t) => {
                name.push_str("Stream");
                t.mangle(ctx, name);
            }
            hir::TypeKind::Struct(fts) => mangle_fields(self, &fts, ctx, name),
            hir::TypeKind::Tuple(ts) => {
                name.push_str("Tuple");
                ts.iter().for_each(|t| t.mangle(ctx, name));
            }
            hir::TypeKind::Vector(t) => {
                name.push_str("Vec");
                t.mangle(ctx, name);
            }
            hir::TypeKind::By(t0, t1) => {
                name.push_str("By");
                t0.mangle(ctx, name);
                t1.mangle(ctx, name);
            }
            hir::TypeKind::Unknown => {}
            hir::TypeKind::Err => {}
        }
    }
}

impl hir::Name {
    fn mangle(&self, ctx: &mut Context<'_>, name: &mut String) {
        name.push_str(ctx.info.names.resolve(self.id));
    }
}

impl hir::Path {
    fn mangle(&self, ctx: &mut Context<'_>, name: &mut String) {
        name.push_str(&ctx.info.resolve_to_names(self.id).join("_"));
    }
}

impl hir::ScalarKind {
    #[rustfmt::skip]
    fn mangle(self, ctx: &mut Context<'_>, name: &mut String) {
        let s = match self {
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
        };
        name.push_str(s);
    }
}
