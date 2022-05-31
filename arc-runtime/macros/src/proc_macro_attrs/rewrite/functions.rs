use crate::get_metas;
use crate::has_meta_key;
use crate::new_id;
use proc_macro as pm;
use syn::visit_mut::VisitMut;

pub(crate) fn rewrite(attr: syn::AttributeArgs, mut item: syn::ItemFn) -> pm::TokenStream {
    item.sig.generics.params.iter_mut().for_each(|p| {
        if let syn::GenericParam::Type(p) = p {
            p.bounds.push(syn::parse_quote!(Data));
        }
    });
    Visitor::default().visit_item_fn_mut(&mut item);
    let (ids, tys): (Vec<_>, Vec<_>) = item
        .sig
        .inputs
        .iter()
        .map(|x| match x {
            syn::FnArg::Receiver(_) => unreachable!(),
            syn::FnArg::Typed(t) => (&t.pat, &t.ty),
        })
        .unzip();

    let id = new_id(format!("_{}", item.sig.ident));
    item.attrs.push(syn::parse_quote!(#[allow(unused_parens)]));
    let mut wrapper_item = item.clone();
    if wrapper_item.sig.asyncness.is_some() {
        wrapper_item.block = syn::parse_quote!({ #id(#(#ids,)* ctx).await });
    }
    let ctx = if has_meta_key("generic", &get_metas(&attr)) {
        quote::quote!(Context<impl Execute>)
    } else {
        quote::quote!(Context<Task>)
    };
    wrapper_item.sig.inputs = syn::parse_quote!((#(#ids,)*) : (#(#tys,)*), mut ctx: #ctx);
    item.sig.ident = id;
    item.sig.inputs.push(syn::parse_quote!(mut ctx: #ctx));
    quote::quote!(
        #wrapper_item
        #item
    )
    .into()
}

pub(crate) struct Visitor {
    scopes: Vec<Scope>,
    roots: Vec<syn::Ident>,
}

struct Scope {
    unroots: Vec<syn::Ident>,
    kind: ScopeKind,
}

impl Scope {
    fn new(kind: ScopeKind) -> Self {
        Self {
            unroots: Vec::new(),
            kind,
        }
    }
}

enum ScopeKind {
    Block,
    FunctionBody,
    LoopBody,
    IfBranch,
}

impl Default for Visitor {
    fn default() -> Self {
        Self {
            scopes: vec![],
            roots: vec![],
        }
    }
}

impl VisitMut for Visitor {
    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        if !i.block.stmts.is_empty() {
            self.scopes.push(Scope::new(ScopeKind::FunctionBody));
            self.visit_block_mut(&mut i.block);
            self.scopes.pop();
        }
    }

    // Every function call must pass an implicit context parameter.
    fn visit_expr_call_mut(&mut self, i: &mut syn::ExprCall) {
        i.args.push(syn::parse_quote!(ctx));
        syn::visit_mut::visit_expr_call_mut(self, i);
    }

    // Every identifier pattern introduces a new Gc<T> root.
    fn visit_pat_ident_mut(&mut self, i: &mut syn::PatIdent) {
        let outmost = self.scopes.last_mut().unwrap();
        self.roots.push(i.ident.clone());
        outmost.unroots.push(i.ident.clone());
    }

    fn visit_expr_block_mut(&mut self, i: &mut syn::ExprBlock) {
        if !i.block.stmts.is_empty() {
            self.scopes.push(Scope::new(ScopeKind::Block));
            syn::visit_mut::visit_expr_block_mut(self, i);
            self.scopes.pop();
        }
    }

    fn visit_expr_loop_mut(&mut self, i: &mut syn::ExprLoop) {
        if !i.body.stmts.is_empty() {
            self.scopes.push(Scope::new(ScopeKind::LoopBody));
            syn::visit_mut::visit_expr_loop_mut(self, i);
            self.scopes.pop();
        }
    }

    fn visit_expr_if_mut(&mut self, i: &mut syn::ExprIf) {
        syn::visit_mut::visit_expr_mut(self, &mut i.cond);
        if !i.then_branch.stmts.is_empty() {
            self.scopes.push(Scope::new(ScopeKind::IfBranch));
            syn::visit_mut::visit_block_mut(self, &mut i.then_branch);
            self.scopes.pop();
        }
        if let Some((_, else_branch)) = &mut i.else_branch {
            match else_branch.as_mut() {
                syn::Expr::Block(b) => {
                    if !b.block.stmts.is_empty() {
                        self.scopes.push(Scope::new(ScopeKind::IfBranch));
                        syn::visit_mut::visit_block_mut(self, &mut b.block);
                        self.scopes.pop();
                    }
                }
                syn::Expr::If(i) => syn::visit_mut::visit_expr_if_mut(self, i),
                _ => unreachable!(),
            }
        }
    }

    // Every block is garbage collected.
    fn visit_block_mut(&mut self, i: &mut syn::Block) {
        let unroot = |stmts: &mut Vec<syn::Stmt>, unroots: &Vec<syn::Ident>| {
            stmts.extend(
                unroots
                    .iter()
                    .map(|id| syn::parse_quote!(#id.unroot(ctx.heap);)),
            );
        };
        let root = |stmts: &mut Vec<syn::Stmt>, roots: &Vec<syn::Ident>| {
            stmts.extend(
                roots
                    .iter()
                    .map(|id| syn::parse_quote!(#id.root(ctx.heap);)),
            );
        };
        // This code roots all the patterns bound for each statement.
        let mut stmts = Vec::new();
        // Add statements to the block and root local variables
        for mut stmt in i.stmts.drain(..i.stmts.len() - 1) {
            self.visit_stmt_mut(&mut stmt);
            match &stmt {
                syn::Stmt::Expr(syn::Expr::Break(_)) => {
                    for scope in self.scopes.iter().rev() {
                        unroot(&mut stmts, &scope.unroots);
                        if let ScopeKind::LoopBody = scope.kind {
                            stmts.push(stmt);
                            return;
                        }
                    }
                }
                syn::Stmt::Expr(syn::Expr::Return(_)) => {
                    for scope in self.scopes.iter().rev() {
                        unroot(&mut stmts, &scope.unroots);
                        if let ScopeKind::FunctionBody = scope.kind {
                            stmts.push(stmt);
                            return;
                        }
                    }
                }
                _ => {
                    stmts.push(stmt);
                    root(&mut stmts, &self.roots);
                    self.roots.clear();
                }
            }
        }
        unroot(&mut stmts, &self.scopes.last().unwrap().unroots);
        let mut stmt = i.stmts.pop().unwrap();
        self.visit_stmt_mut(&mut stmt);
        stmts.push(stmt);
        // Unroot all variables in this scope
        i.stmts = stmts;
    }

    // We have this to make sure we visit the let-expression before the let-pattern
    fn visit_local_mut(&mut self, i: &mut syn::Local) {
        if let Some((_, init)) = &mut i.init {
            syn::visit_mut::visit_expr_mut(self, &mut *init);
        } else {
            panic!("Local without initializer");
        }
        syn::visit_mut::visit_pat_mut(self, &mut i.pat);
    }
}
