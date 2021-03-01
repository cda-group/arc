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

impl hir::Path {
    pub(crate) fn lower_state(&self, ctx: &mut Context<'_>) -> Option<(Tokens, Tokens)> {
        let item = ctx.hir.defs.get(self).unwrap();
        map!(&item.kind, hir::ItemKind::State(_)).map(|item| item.lower_state(ctx))
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
