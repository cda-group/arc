use diagnostics::Diagnostics;
use hir::*;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use name_gen::NameGen;
use stack::Stack;

#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) stack: Stack<ScopeKind, ExprDecl, TypeDecl>,
    next_type_uid: NameGen,
    next_row_uid: NameGen,
    subst: Vector<(Name, Type)>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
pub(crate) enum ScopeKind {
    Top,
    Block,
    While,
    For,
    Loop(Type),
    Def(Type),
    Fun(Type),
    Enum,
    Type,
    Arm,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprDecl {
    Def(Info, Type, Vector<Generic>, Vector<Generic>),
    Var(Info, Type),
    Val(Info, Type),
    Variant(Info, Name, usize),
}

#[derive(Clone, Debug)]
pub(crate) enum TypeDecl {
    Enum(Info, Vector<Generic>, Vector<(Name, Type)>),
    Bit(Info, Vector<Generic>),
    Generic(Info),
}

impl Default for Context {
    fn default() -> Self {
        Self {
            stack: Stack::new(ScopeKind::Top),
            subst: vector![],
            next_type_uid: NameGen::new("u"),
            next_row_uid: NameGen::new("r"),
            diagnostics: Diagnostics::default(),
        }
    }
}

impl Context {
    pub fn new() -> Context {
        Self::default()
    }

    pub(crate) fn return_type(&self) -> Type {
        self.stack
            .iter()
            .find_map(|scope| match &scope.kind {
                ScopeKind::Fun(ty) => Some(ty),
                _ => None,
            })
            .cloned()
            .unwrap()
    }

    pub(crate) fn break_type(&self) -> Type {
        self.stack
            .iter()
            .find_map(|scope| match &scope.kind {
                ScopeKind::Loop(ty) => Some(ty),
                _ => None,
            })
            .cloned()
            .unwrap()
    }

    pub(crate) fn fresh_t(&mut self) -> Type {
        TVar(self.next_type_uid.fresh()).into()
    }

    pub(crate) fn fresh_ts(&mut self, n: i32) -> Vector<Type> {
        (0..n).into_iter().fold(vector![], |mut tys, _| {
            let ty = self.fresh_t();
            tys.push_back(ty);
            tys
        })
    }

    pub(crate) fn fresh_r(&mut self) -> Type {
        TVar(self.next_row_uid.fresh()).into()
    }

    pub(crate) fn get_subst(&self) -> Vector<(Name, Type)> {
        self.subst.clone()
    }

    pub(crate) fn set_subst(&mut self, subst: Vector<(Name, Type)>) {
        self.subst = subst;
    }
}
