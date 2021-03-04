use arc_script_core_shared::get;
use arc_script_core_shared::map;
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
    pub(crate) fn lower_method(&self, ctx: &mut Context<'_>) -> Option<Tokens> {
        let item = ctx.hir.defs.get(self).unwrap();
        map!(&item.kind, hir::ItemKind::Fun(_)).map(|item| item.lower(ctx))
    }
}

impl hir::State {
    pub(crate) fn lower_state(
        &self,
        backend: &Tokens,
        ctx: &mut Context<'_>,
    ) -> (Tokens, Tokens, Tokens) {
        let ty = ctx.info.types.resolve(self.tv);
        let name = self.path.lower(ctx);
        let init = self.init.lower(ctx);
        let ty = match ty.kind {
            hir::TypeKind::Map(t0, t1) => {
                let t0 = t0.lower(ctx);
                let t1 = t1.lower(ctx);
                quote!(arc_script::arcorn::Map<#t0, #t1, #backend>)
            }
            hir::TypeKind::Vector(t0) => {
                let t0 = t0.lower(ctx);
                quote!(arc_script::arcorn::Appender<#t0, #backend>)
            }
            hir::TypeKind::Set(t0) => {
                let t0 = t0.lower(ctx);
                quote!(arc_script::arcorn::Set<#t0, #backend>)
            }
            _ => {
                let t0 = self.tv.lower(ctx);
                quote!(arc_script::arcorn::Value<#t0, #backend>)
            }
        };
        let state_decl = quote!(#name : #ty);
        let state_init = quote!(#name : #init);
        let state_id = quote!(#name);
        (state_decl, state_init, state_id)
    }
}
