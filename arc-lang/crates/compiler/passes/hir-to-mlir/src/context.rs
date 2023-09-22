use diagnostics::Diagnostics;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use mlir::*;
use name_gen::NameGen;
use stack::Stack;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) next_expr_name: NameGen,
    pub(crate) stack: Stack<ScopeKind, ExprDecl, TypeDecl>,
    pub diagnostics: Diagnostics,
    pub(crate) items: Vector<Item>,
}

#[derive(Clone, Debug)]
pub(crate) enum ScopeKind {
    Top,
    Block,
    While,
    Def,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprDecl {
    Def,
    Bif(Name, Type),
    Var(Type),
    Variant(Info, Name),
}

#[derive(Clone, Debug)]
pub(crate) enum TypeDecl {
    Enum(Vector<(Name, Type)>),
    Native(Name),
    Adt(Name),
}

impl Default for Context {
    fn default() -> Self {
        Self {
            next_expr_name: NameGen::new("x"),
            stack: Stack::new(ScopeKind::Top),
            diagnostics: Diagnostics::default(),
            items: vector![],
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn fresh_expr_name(&mut self) -> Name {
        self.next_expr_name.fresh()
    }

    pub(crate) fn add_item(&mut self, item: Item) {
        self.items.push_back(item);
    }
}
