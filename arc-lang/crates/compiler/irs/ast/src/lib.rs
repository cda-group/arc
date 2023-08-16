#![allow(unused)]

use im_rc::Vector;
use info::Info;
use lexer::tokens::Token;
use std::rc::Rc;

pub mod ops;

pub use Splice::*;
#[derive(Clone, Debug)]
pub enum Splice {
    SName(Name),
    SBlock(Block),
}

pub type Name = String;
pub type Arm = (Pattern, Expr);

pub type Index = i32;

pub use ExprField::*;
#[derive(Clone, Debug)]
pub enum ExprField {
    FName(Name, Option<Expr>),
    FExpr(Expr, Name),
}

pub use Body::*;
#[derive(Clone, Debug)]
pub enum Body {
    BBlock(Block),
    BExpr(Expr),
}

#[derive(Clone, Debug)]
pub struct Block {
    pub ss: Vector<Stmt>,
    pub e: Option<Expr>,
    pub info: Info,
}

pub type Generic = Name;

pub type Meta = Vector<Attr>;

#[derive(Clone, Debug)]
pub struct Attr {
    pub x: Name,
    pub c: Option<Const>,
    pub info: Info,
}
pub type Bound = (Name, Vector<Type>);

pub use UseSuffix::*;
#[derive(Clone, Debug)]
pub enum UseSuffix {
    UAlias(Name),
    UGlob,
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub info: Info,
    pub kind: Rc<PatternKind>,
}

pub use PatternKind::*;
#[derive(Clone, Debug)]
pub enum PatternKind {
    PParen(Pattern),
    PIgnore,
    POr(Pattern, Pattern),
    PTypeAnnot(Pattern, Type),
    PRecord(Vector<(Name, Option<Pattern>)>),
    PRecordConcat(Pattern, Pattern),
    PTuple(Vector<Pattern>),
    PArray(Vector<Pattern>),
    PArrayConcat(Pattern, Pattern),
    PConst(Const),
    PName(Name),
    PVariantRecord(Name, Vector<(Name, Option<Pattern>)>),
    PVariantTuple(Name, Vector<Pattern>),
    PError,
}

#[derive(Clone, Debug)]
pub struct Type {
    pub info: Info,
    pub kind: Rc<TypeKind>,
}

pub use TypeKind::*;
#[derive(Clone, Debug)]
pub enum TypeKind {
    TParen(Type),
    TFun(Vector<Type>, Type),
    TTuple(Vector<Type>),
    TRecord(Vector<(Name, Type)>),
    TRecordConcat(Type, Type),
    TName(Name, Vector<Type>),
    TArray(Type, Option<i32>),
    TArrayConcat(Type, Type),
    TUnit,
    TNever,
    TIgnore,
    TError,
}

#[derive(Clone, Debug)]
pub struct Binop {
    pub token: Token,
    pub kind: BinopKind,
}

pub use BinopKind::*;
#[derive(Clone, Copy, Debug)]
pub enum BinopKind {
    BAdd,
    BAnd,
    BDiv,
    BEq,
    BGeq,
    BGt,
    BLeq,
    BLt,
    BMul,
    BNeq,
    BOr,
    BSub,
    BRExc,
    BRInc,
}

impl BinopKind {
    pub fn with(self, token: Token) -> Binop {
        Binop { token, kind: self }
    }
}

#[derive(Clone, Debug)]
pub struct Unop {
    pub token: Token,
    pub kind: UnopKind,
}

pub use UnopKind::*;
#[derive(Clone, Copy, Debug)]
pub enum UnopKind {
    UNot,
    UNeg,
    UPos,
}

impl UnopKind {
    pub fn with(self, token: Token) -> Unop {
        Unop { token, kind: self }
    }
}

pub use Lit::*;
#[derive(Clone, Debug)]
pub enum Lit {
    LInt(i32, Option<Name>),
    LFloat(f32, Option<Name>),
    LBool(bool),
    LString(String),
    LUnit,
    LChar(char),
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

pub use StmtKind::*;
#[derive(Clone, Debug)]
pub struct Stmt {
    pub info: Info,
    pub kind: StmtKind,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    SDef(
        Meta,
        Name,
        Vector<Generic>,
        Vector<Pattern>,
        Option<Type>,
        Vector<Bound>,
        Body,
    ),
    SEnum(Meta, Name, Vector<Generic>, Vector<Bound>, Vector<Variant>),
    SType(Meta, Name, Vector<Generic>, Type),
    SNoop,
    SVal(Pattern, Expr),
    SVar(Pattern, Expr),
    SExpr(Expr),
    SInject(String, String),
    // Internal
    SBuiltinDef(
        Meta,
        Name,
        Vector<Generic>,
        Vector<Type>,
        Option<Type>,
        Vector<Bound>,
    ),
    SBuiltinType(Meta, Name, Vector<Generic>, Vector<Bound>),
    SBuiltinClass(Meta, Name, Vector<Generic>, Vector<Bound>),
    SBuiltinInstance(Meta, Name, Vector<Generic>, Vector<Bound>, Type),
}

impl StmtKind {
    pub fn with(self, info: Info) -> Stmt {
        Stmt { info, kind: self }
    }
}

pub use Variant::*;
#[derive(Clone, Debug)]
pub enum Variant {
    VUnit(Name),
    VRecord(Name, Vector<(Name, Type)>),
    VTuple(Name, Vector<Type>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub info: Info,
    pub kind: Rc<ExprKind>,
}

impl Expr {
    pub fn kind(&self) -> ExprKind {
        (*self.kind).clone()
    }
}

impl ExprKind {
    pub fn with(self, info: Info) -> Expr {
        Expr {
            info,
            kind: Rc::new(self),
        }
    }
}

impl PatternKind {
    pub fn with(self, info: Info) -> Pattern {
        Pattern {
            info,
            kind: Rc::new(self),
        }
    }
}

impl TypeKind {
    pub fn with(self, info: Info) -> Type {
        Type {
            info,
            kind: Rc::new(self),
        }
    }
}

impl Type {
    pub fn kind(&self) -> TypeKind {
        (*self.kind).clone()
    }
}

impl Pattern {
    pub fn kind(&self) -> PatternKind {
        (*self.kind).clone()
    }
}

pub use ExprKind::*;
#[derive(Clone, Debug)]
pub enum ExprKind {
    EParen(Expr),
    EQuery(Pattern, Expr, Vector<QueryStmt>),
    EFunCall(Expr, Vector<Expr>),
    EFunReturn(Option<Expr>),
    EIfElse(Expr, Block, Option<Block>),
    ELit(Lit),
    ELoop(Block),
    ELoopBreak(Option<Expr>),
    ELoopContinue,
    ERecord(Vector<ExprField>),
    ERecordAccess(Expr, Name),
    ERecordAccessMulti(Expr, Vector<Name>),
    ERecordConcat(Expr, Expr),
    ETypeAnnot(Expr, Type),
    EVariantRecord(Name, Vector<ExprField>),
    EDo(Block),
    EArray(Vector<Expr>),
    EArrayConcat(Expr, Expr),
    EArrayAccess(Expr, Expr),
    EBinop(Expr, Binop, Expr),
    EMut(Expr, Option<Binop>, Expr),
    EFor(Pattern, Expr, Block),
    EFun(Vector<Pattern>, Option<Type>, Body),
    EMatch(Expr, Vector<Arm>),
    EMethodCall(Expr, Name, Vector<Type>, Vector<Expr>),
    EName(Name, Vector<Type>),
    EThrow(Expr),
    ETry(Block, Vector<Arm>, Option<Block>),
    ETuple(Vector<Expr>),
    ETupleAccess(Expr, Index),
    EUnop(Unop, Expr),
    EWhile(Expr, Block),
    EError,
}

pub use QueryStmt::*;
#[derive(Clone, Debug)]
pub enum QueryStmt {
    QFrom(Pattern, Expr),
    QWhere(Expr),
    QWith(Pattern, Expr),
    QJoinOn(Pattern, Expr, Expr),
    QJoinOver(Pattern, Expr, Expr),
    QJoinOverOn(Pattern, Expr, Expr, Expr, Expr, Vector<QueryStmt>),
    QGroup(Expr, Vector<QueryStmt>, Option<Name>),
    QCompute(Expr, Option<Expr>, Option<Name>),
    QOver(Expr, Vector<QueryStmt>, Option<Name>),
    QRoll(Expr, Option<Expr>, Option<Name>),
    QSelect(Expr),
    QUnion(Expr),
    QInto(Expr),
    QVal(Pattern, Expr),
    QOrder(Expr, Order),
}

pub use Order::*;
#[derive(Clone, Debug)]
pub enum Order {
    OAsc,
    ODesc,
}

impl Stmt {
    pub fn kind(&self) -> StmtKind {
        self.kind.clone()
    }
}

impl Block {
    pub fn new(ss: Vector<Stmt>, e: Option<Expr>, info: Info) -> Self {
        Self { ss, e, info }
    }
}
