#![allow(unused)]
use std::collections::HashMap;

use hir::*;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use name_gen::NameGen;
use stack::Stack;

use value::Value;

use crate::definitions::Bifs;

#[derive(Debug)]
pub struct Context {
    pub(crate) stack: Stack<(), ExprDecl, TypeDecl>,
    pub(crate) bifs: Bifs,
    next_stream_name: NameGen,
    pub ss: Vector<hir::Stmt>,
    pub ctx7: hir_reachable::context::Context,
    pub ctx8: hir_to_mlir::context::Context,
    pub ctx9: hir_to_rust::context::Context,
    pub ctx10: build::context::Context,
    pub ctx11: Option<kafka::context::Context>,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprDecl {
    Def(Meta, Vector<Generic>, Vector<Pattern>, Type, Block),
    Bif(Meta, Vector<Generic>, Vector<Type>, Type),
    Var(Value),
    Val(Value),
    Variant(Name, Vector<Generic>),
}

#[derive(Clone, Debug)]
pub(crate) enum TypeDecl {
    Enum(Vector<Generic>, Vector<(Name, Type)>),
    Bit(Vector<Generic>),
    Generic(Type),
}

impl Default for Context {
    fn default() -> Self {
        Context {
            stack: Stack::new(()),
            next_stream_name: NameGen::new("s"),
            bifs: Bifs::new(),
            ss: Vector::new(),
            ctx7: Default::default(),
            ctx8: Default::default(),
            ctx9: Default::default(),
            ctx10: Default::default(),
            ctx11: Default::default(),
        }
    }
}

impl Context {
    pub fn new() -> Context {
        Self::default()
    }

    pub fn find_val(&self, x: &Name) -> Value {
        self.stack
            .find_expr_decl(x)
            .and_then(|decl| match decl {
                ExprDecl::Var(v) => Some(v.clone()),
                _ => None,
            })
            .unwrap()
    }

    pub fn new_stream_name(&mut self) -> Name {
        self.next_stream_name.fresh()
    }
}
