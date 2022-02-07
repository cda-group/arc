use crate::rust::lower::hir::Context;
use crate::hir;
use crate::hir::Name;
use crate::hir::Type;

use proc_macro2 as pm2;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::VecMap;

/// Mangles an anonymous-struct into a nominal-struct, e.g.
///
/// `{a_i32_b: i32}   => Struct7a_i32_b3i32`
/// `{a: i32, b: i32} => Struct1a3i321b3i32`
///
impl Type {
    pub(crate) fn mangle_to_ident(self, ctx: &mut Context<'_>) -> syn::Ident {
        let mut name = String::new();
        let fts = get!(ctx.types.resolve(self), hir::TypeKind::Struct(fs));
        self.mangle_fields(&fts, &mut name, ctx);
        let span = pm2::Span::call_site();
        syn::Ident::new(&name, span)
    }

    fn mangle_fields(self, fts: &VecMap<Name, Type>, name: &mut String, ctx: &mut Context<'_>) {
        let root = ctx.types.root(self).into();
        if let Some(ident) = ctx.mangled_names.get(&root) {
            name.push_str(ident);
        } else {
            let mut inner = String::new();
            // Mangle the struct
            inner.push_str("Struct");
            fts.iter().for_each(|(x, t)| {
                let len = ctx.names.resolve(x).len();
                inner.push_str(&format!("{}", len));
                x.mangle(&mut inner, ctx);
                t.mangle(&mut inner, ctx);
            });
            inner.push_str("End");
            ctx.mangled_names.insert(root, inner.clone());
            name.push_str(&inner);
        }
    }
}

trait Mangle {
    #[allow(clippy::style)]
    fn mangle(&self, name: &mut String, ctx: &mut Context<'_>);
}

/// Macro for implementing the `Declare` trait.
macro_rules! mangle {
    {
        [$node:ident, $name:ident, $ctx:ident]
        $($ty:path => $expr:expr ,)*
    } => {
        $(
            impl Mangle for $ty {
                fn mangle(&self, $name: &mut String, $ctx: &mut Context<'_>) {
                    let $node = self;
                    $expr;
                }
            }
        )*
    };
}

mangle! {
    [node, name, ctx]

    hir::Type => match ctx.types.resolve(*node) {
        hir::TypeKind::Array(t, _s) => {
            name.push_str("Array");
            t.mangle(name, ctx);
            crate::todo!();
        }
        hir::TypeKind::Fun(ts, t) => {
            name.push_str("Fun");
            ts.iter().for_each(|t| t.mangle(name, ctx));
            t.mangle(name, ctx);
        }
        hir::TypeKind::Nominal(x) => x.mangle(name, ctx),
        hir::TypeKind::Scalar(s) => s.mangle(name, ctx),
        hir::TypeKind::Stream(t) => {
            name.push_str("Stream");
            t.mangle(name, ctx);
        }
        hir::TypeKind::Struct(fts) => node.mangle_fields(&fts, name, ctx),
        hir::TypeKind::Tuple(ts) => {
            name.push_str("Tuple");
            ts.iter().for_each(|t| t.mangle(name, ctx));
        }
        hir::TypeKind::Unknown(_) => {}
        hir::TypeKind::Err => {}
    },
    hir::Name => name.push_str(ctx.names.resolve(node)),
    hir::Path => node.id.mangle(name, ctx),
    hir::PathId => name.push_str(&ctx.resolve_to_names(node).join("_")),
    hir::ScalarKind => {
        let s = match node {
            hir::ScalarKind::Bool      => "bool",
            hir::ScalarKind::Char      => "char",
            hir::ScalarKind::F32       => "f32",
            hir::ScalarKind::F64       => "f64",
            hir::ScalarKind::I8        => "i8",
            hir::ScalarKind::I16       => "i16",
            hir::ScalarKind::I32       => "i32",
            hir::ScalarKind::I64       => "i64",
            hir::ScalarKind::U8        => "u8",
            hir::ScalarKind::U16       => "u16",
            hir::ScalarKind::U32       => "u32",
            hir::ScalarKind::U64       => "u64",
            hir::ScalarKind::Str       => crate::todo!(),
            hir::ScalarKind::Unit      => "Unit",
            hir::ScalarKind::Size      => "usize",
            hir::ScalarKind::DateTime  => "time",
            hir::ScalarKind::Duration  => "duration",
        };
        name.push_str(s);
    },
}
