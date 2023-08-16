use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use std::hash::Hash;
use std::rc::Rc;

pub type Name = String;
pub type Index = i128;
pub type Generic = Name;
pub type Meta = OrdMap<Name, Option<Const>>;

#[derive(Clone, Debug)]
pub struct Item {
    pub info: Info,
    pub kind: ItemKind,
}
pub use ItemKind::*;
#[derive(Clone, Debug)]
pub enum ItemKind {
    IDef(Meta, Name, Vector<Val>, Type, Block),
    IEnum(Meta, Name, Vector<(Name, Type)>),
    IStruct(Name, Vector<(Name, Type)>),
    IError,
}

pub use Const::*;
#[derive(Clone, Debug)]
pub enum Const {
    CInt(i32),
    CFloat(f32),
    CBool(bool),
    CString(String),
    CUnit,
    CChar(char),
}

#[derive(Clone, Debug)]
pub struct Val {
    pub kind: ValKind,
    pub t: Type,
}

pub use ValKind::*;
#[derive(Clone, Debug)]
pub enum ValKind {
    VName(Name),
    VError,
}

impl ValKind {
    pub fn with(self, t: Type) -> Val {
        Val { kind: self, t }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub ss: Vector<Stmt>,
}

impl Block {
    pub fn new(ss: Vector<Stmt>) -> Self {
        Block { ss }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Type {
    pub kind: Rc<TypeKind>,
}

pub use TypeKind::*;
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum TypeKind {
    TFun(Vector<Type>, Type),
    TNominal(Name, Vector<Type>),
    TError,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub vs: Vector<Val>,
    pub kind: StmtKind,
}

pub use StmtKind::*;
#[derive(Clone, Debug)]
pub enum StmtKind {
    SConst(Const),
    SFun(Name),
    SFunCallDirect(Name, Vector<Val>),
    SFunCallIndirect(Val, Vector<Val>),
    SIfElse(Val, Block, Block),
    SStruct(Name, Vector<(Name, Val)>),
    SStructAccess(Val, Name),
    SVariant(Name, Val),
    SVariantAccess(Name, Val),
    SVariantCheck(Name, Val),
    SWhile(Vector<Val>, Vector<Val>, Block, Block),
    SWhileBreak(Vector<Val>),
    SWhileContinue(Vector<Val>),
    SWhileYield(Vector<Val>),
    SBlockResult(Val),
    SFunReturn(Val),
    SError,
}

impl From<TypeKind> for Type {
    fn from(kind: TypeKind) -> Type {
        Type {
            kind: Rc::new(kind),
        }
    }
}

impl ItemKind {
    pub fn with(self, info: Info) -> Item {
        Item { kind: self, info }
    }
}

impl StmtKind {
    pub fn with(self, vs: impl IntoIterator<Item = Val>) -> Stmt {
        Stmt {
            vs: vs.into_iter().collect(),
            kind: self,
        }
    }
}
