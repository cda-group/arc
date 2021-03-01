use arc_script_core_shared::get;
use arc_script_core_shared::Bool;
use arc_script_core_shared::Lower;

use crate::compiler::arcon::from::lower::lowerings::structs;
use crate::compiler::hir;
use crate::compiler::info::Info;

use super::super::Context;

use proc_macro2 as pm2;
use proc_macro2::TokenStream as Tokens;
use quote::quote;

impl hir::Path {
    #[rustfmt::skip]
    pub(crate) fn lower_method(&self, ctx: &mut Context<'_>) -> Option<Tokens> {
        let item = ctx.hir.defs.get(self).unwrap();
        match &item.kind {
            hir::ItemKind::Alias(item)   => None,
            hir::ItemKind::Enum(item)    => None,
            hir::ItemKind::Fun(item)     => Some(item.lower_method(ctx)),
            hir::ItemKind::State(item)   => None,
            hir::ItemKind::Task(item)    => None,
            hir::ItemKind::Extern(item)  => None,
            hir::ItemKind::Variant(item) => None,
        }
    }
}

impl hir::Fun {
    fn lower_method(&self, ctx: &mut Context<'_>) -> Tokens {
        let name = self.path.lower(ctx);
        let rtv = self.rtv.lower(ctx);
        let body = self.body.lower(ctx);
        let params = self.params.iter().map(|p| p.lower(ctx));
        quote! {
            fn #name(&mut self, #(#params),*) -> #rtv {
                #body
            }
        }
    }
}

impl hir::Path {
    #[rustfmt::skip]
    pub(crate) fn lower_state(&self, ctx: &mut Context<'_>) -> Option<(Tokens, Tokens)> {
        let item = ctx.hir.defs.get(self).unwrap();
        match &item.kind {
            hir::ItemKind::Alias(item)   => None,
            hir::ItemKind::Enum(item)    => None,
            hir::ItemKind::Fun(item)     => None,
            hir::ItemKind::State(item)   => Some(item.lower_state(ctx)),
            hir::ItemKind::Task(item)    => None,
            hir::ItemKind::Extern(item)  => None,
            hir::ItemKind::Variant(item) => None,
        }
    }
}

impl hir::State {
    fn lower_state(&self, ctx: &mut Context<'_>) -> (Tokens, Tokens) {
        let name = self.path.lower(ctx);
        let tv = self.tv.lower(ctx);
        let init = self.init.lower(ctx);
        (quote!(#name : #tv), quote!(#name : #init))
    }
}
