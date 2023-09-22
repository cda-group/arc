use diagnostics::Diagnostics;
use im_rc::vector;
use im_rc::HashMap;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use name_gen::NameGen;
use rust::*;
use stack::Stack;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) next_struct_name: NameGen,
    pub(crate) stack: Stack<ScopeKind, ExprDecl, TypeDecl>,
    pub diagnostics: Diagnostics,
    pub(crate) items: Vector<Item>,
    pub(crate) structs: HashMap<Vector<(Name, Type)>, Name>,
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
    Adt(Name),
}

impl Default for Context {
    fn default() -> Self {
        Self {
            next_struct_name: NameGen::new("S"),
            stack: Stack::new(ScopeKind::Top),
            diagnostics: Diagnostics::default(),
            items: vector![],
            structs: HashMap::new(),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn fresh_struct_name(&mut self) -> Name {
        self.next_struct_name.fresh()
    }

    pub(crate) fn add_item(&mut self, item: Item) {
        self.items.push_back(item);
    }
}
