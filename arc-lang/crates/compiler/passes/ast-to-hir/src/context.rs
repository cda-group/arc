use diagnostics::Diagnostics;
use hir::*;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use name_gen::NameGen;
use stack::Stack;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Context {
    next_type_var_uid: NameGen,
    pub(crate) stack: Stack<ScopeKind, ExprDecl, TypeDecl>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
pub(crate) struct Scope {
    kind: ScopeKind,
    expr_namespace: Vec<(ast::Name, ExprDecl)>, // BIFs, Functions, variables, parameters
    type_namespace: Vec<(ast::Name, TypeDecl)>, // BITs, Type aliases, generics, enums
}

#[derive(Clone, Debug)]
pub(crate) enum ScopeKind {
    Top,
    Block,
    While,
    For,
    Loop,
    Def,
    Fun,
    Enum,
    Type,
    Arm,
    Query,
    QuerySelect,
}

#[derive(Clone, Debug)]
pub enum ExprDecl {
    Def(Info, usize, Vector<Generic>),
    Var(Info),
    Val(Info),
    Variant(Info, Name, Vector<Generic>),
}

impl ExprDecl {
    pub fn info(&self) -> Info {
        match self {
            ExprDecl::Def(info, _, _) => *info,
            ExprDecl::Var(info) => *info,
            ExprDecl::Val(info) => *info,
            ExprDecl::Variant(info, _, _) => *info,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TypeDecl {
    Enum(Info, Vector<Generic>, Vector<(Name, Type)>),
    Type(Info, Vector<Generic>, Type),
    TypeArg(Info, Type),
    Bit(Info, Vector<Generic>),
    Generic(Info),
}

impl TypeDecl {
    pub fn info(&self) -> Info {
        match self {
            TypeDecl::Enum(info, _, _) => *info,
            TypeDecl::Type(info, _, _) => *info,
            TypeDecl::TypeArg(info, _) => *info,
            TypeDecl::Bit(info, _) => *info,
            TypeDecl::Generic(info) => *info,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Context {
            next_type_var_uid: NameGen::new("t"),
            stack: Stack::new(ScopeKind::Top),
            diagnostics: Diagnostics::default(),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_inside_loop(&self) -> bool {
        for scope in self.stack.iter() {
            match scope.kind {
                ScopeKind::Top => break,
                ScopeKind::Block => continue,
                ScopeKind::Loop => return true,
                ScopeKind::While => return true,
                ScopeKind::For => return true,
                ScopeKind::Def => break,
                ScopeKind::Fun => break,
                ScopeKind::Enum => continue,
                ScopeKind::Type => continue,
                ScopeKind::Arm => continue,
                ScopeKind::Query => continue,
                ScopeKind::QuerySelect => continue,
            }
        }
        false
    }

    pub(crate) fn is_inside_infinite_loop(&self) -> bool {
        for scope in self.stack.iter() {
            match scope.kind {
                ScopeKind::Top => break,
                ScopeKind::Block => continue,
                ScopeKind::Loop => return true,
                ScopeKind::While => break,
                ScopeKind::For => break,
                ScopeKind::Def => break,
                ScopeKind::Fun => break,
                ScopeKind::Enum => continue,
                ScopeKind::Type => continue,
                ScopeKind::Arm => continue,
                ScopeKind::Query => continue,
                ScopeKind::QuerySelect => continue,
            }
        }
        false
    }

    pub(crate) fn is_inside_function(&self) -> bool {
        for scope in self.stack.iter() {
            match scope.kind {
                ScopeKind::Top => break,
                ScopeKind::Block => continue,
                ScopeKind::Loop => continue,
                ScopeKind::While => continue,
                ScopeKind::For => continue,
                ScopeKind::Def => return true,
                ScopeKind::Fun => return true,
                ScopeKind::Enum => continue,
                ScopeKind::Type => continue,
                ScopeKind::Arm => continue,
                ScopeKind::Query => continue,
                ScopeKind::QuerySelect => continue,
            }
        }
        false
    }

    // from (x, y) in source(..) # x, y
    // select {x}                # x
    // where f(x)                # x
    // select {x, r:g(x)}        # x, r
    // into source(..)           # x, r
    pub(crate) fn query_variables(&self) -> Vector<(ast::Name, ExprDecl)> {
        let mut variables = Vector::new();
        let mut scopes = self.stack.iter();
        while let Some(scope) = scopes.next() {
            match scope.kind {
                ScopeKind::Top => break,
                ScopeKind::Block => continue,
                ScopeKind::Loop => continue,
                ScopeKind::While => continue,
                ScopeKind::For => continue,
                ScopeKind::Def => continue,
                ScopeKind::Fun => continue,
                ScopeKind::Enum => continue,
                ScopeKind::Type => continue,
                ScopeKind::Arm => continue,
                ScopeKind::Query => {
                    // variables.extend(scope.expr_namespace.clone());
                    continue;
                }
                ScopeKind::QuerySelect => {
                    // variables.extend(scope.expr_namespace.clone());
                    continue;
                }
            }
        }
        variables
    }

    pub(crate) fn typed<T>(&mut self, f: impl FnOnce(Type) -> T) -> T {
        f(self.new_type_var())
    }

    pub(crate) fn new_type_var(&mut self) -> Type {
        TVar(self.next_type_var_uid.fresh()).into()
    }
}
