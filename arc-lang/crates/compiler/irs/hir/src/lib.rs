#![allow(unused)]
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use std::rc::Rc;
use utils::VectorUtils;

pub type Arm = (Pattern, Block);

pub type Name = String;
pub type Index = i32;
pub type Generic = Name;
pub type Meta = OrdMap<Name, Option<Const>>;

#[derive(Clone, Debug)]
pub struct Block {
    pub ss: Vector<Stmt>,
    pub e: Expr,
    pub info: Info,
}

impl Block {
    pub fn new(ss: Vector<Stmt>, e: Expr, info: Info) -> Self {
        Block { ss, e, info }
    }
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub t: Type,
    pub info: Info,
    pub kind: Rc<PatternKind>,
}

pub use PatternKind::*;
#[derive(Clone, Debug)]
pub enum PatternKind {
    PIgnore,
    POr(Pattern, Pattern),
    PNoop(Pattern),
    PRecord(Vector<(Name, Pattern)>),
    PRecordConcat(Pattern, Pattern),
    PArray(Vector<Pattern>),
    PArrayConcat(Pattern, Pattern),
    PTuple(Vector<Pattern>),
    PConst(Const),
    PVal(Name),
    PVar(Name),
    PVariant(Name, Vector<Type>, Name, Pattern),
    PError,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Type {
    pub kind: Rc<TypeKind>,
}

pub use TypeKind::*;
#[derive(Clone, Debug)]
pub enum TypeKind {
    TFun(Vector<Type>, Type),
    TTuple(Vector<Type>, bool),
    TRecord(Type),
    TNominal(Name, Vector<Type>),
    TAlias(Info, Info, Type),
    TRowEmpty,
    TRowExtend((Name, Type), Type),
    TRecordConcat(Type, Type),
    TGeneric(Name),
    TArray(Type, Option<i32>),
    TArrayConcat(Type, Type),
    TUnit,
    TNever,
    TVar(Name),
    TError,
}

impl Ord for TypeKind {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Less)
    }
}

impl PartialOrd for TypeKind {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (TFun(ts0, t0), TFun(ts1, t1)) => ts0.partial_cmp(ts1).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    t0.partial_cmp(t1)
                } else {
                    Some(o)
                }
            }),
            (TTuple(ts, c0), TTuple(ts1, c1)) => ts.partial_cmp(ts1).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    c0.partial_cmp(c1)
                } else {
                    Some(o)
                }
            }),
            (TRecord(t0), TRecord(t1)) => t0.partial_cmp(t1),
            (TNominal(x0, ts0), TNominal(x1, ts1)) => x0.partial_cmp(x1).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    ts0.partial_cmp(ts1)
                } else {
                    Some(o)
                }
            }),
            (TAlias(_, _, t0), TAlias(_, _, t1)) => t0.partial_cmp(t1),
            (TRowEmpty, TRowEmpty) => Some(std::cmp::Ordering::Equal),
            (TRowExtend(xt0, t0), TRowExtend(xt1, t1)) => xt0.partial_cmp(xt1).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    t0.partial_cmp(t1)
                } else {
                    Some(o)
                }
            }),
            (TRecordConcat(t00, t01), TRecordConcat(t10, t11)) => {
                t00.partial_cmp(t10).and_then(|o| {
                    if o == std::cmp::Ordering::Equal {
                        t01.partial_cmp(t11)
                    } else {
                        Some(o)
                    }
                })
            }
            (TGeneric(x0), TGeneric(x1)) => x0.partial_cmp(x1),
            (TArray(t0, n0), TArray(t1, n1)) => t0.partial_cmp(t1).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    n0.partial_cmp(n1)
                } else {
                    Some(o)
                }
            }),
            (TArrayConcat(t0, t1), TArrayConcat(t2, t3)) => t0.partial_cmp(t2).and_then(|o| {
                if o == std::cmp::Ordering::Equal {
                    t1.partial_cmp(t3)
                } else {
                    Some(o)
                }
            }),
            (TUnit, TUnit) => Some(std::cmp::Ordering::Equal),
            (TNever, TNever) => Some(std::cmp::Ordering::Equal),
            (TVar(x0), TVar(x1)) => x0.partial_cmp(x1),
            (TError, TError) => Some(std::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

impl PartialEq for TypeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TFun(ts0, t0), TFun(ts1, t1)) => ts0 == ts1 && t0 == t1,
            (TTuple(ts, c0), TTuple(ts1, c1)) => ts == ts1 && c0 == c1,
            (TRecord(t0), TRecord(t1)) => t0 == t1,
            (TNominal(x0, ts0), TNominal(x1, ts1)) => x0 == x1 && ts0 == ts1,
            (TAlias(_, _, t0), TAlias(_, _, t1)) => t0 == t1,
            (TRowEmpty, TRowEmpty) => true,
            (TRowExtend(xt0, t0), TRowExtend(xt1, t1)) => xt0 == xt1 && t0 == t1,
            (TRecordConcat(t00, t01), TRecordConcat(t10, t11)) => t00 == t10 && t01 == t11,
            (TGeneric(x0), TGeneric(x1)) => x0 == x1,
            (TArray(t0, n0), TArray(t1, n1)) => t0 == t1 && n0 == n1,
            (TUnit, TUnit) => true,
            (TNever, TNever) => true,
            (TVar(x0), TVar(x1)) => x0 == x1,
            (TError, TError) => true,
            _ => false,
        }
    }
}

impl Eq for TypeKind {}

impl std::hash::Hash for TypeKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            TFun(a, b) => {
                state.write_u8(0);
                a.hash(state);
                b.hash(state);
            }
            TTuple(a, b) => {
                state.write_u8(1);
                a.hash(state);
                b.hash(state);
            }
            TRecord(a) => {
                state.write_u8(2);
                a.hash(state);
            }
            TNominal(a, b) => {
                state.write_u8(3);
                a.hash(state);
                b.hash(state);
            }
            TAlias(a, b, c) => {
                c.hash(state);
            }
            TRowEmpty => {
                state.write_u8(5);
            }
            TRowExtend(a, b) => {
                state.write_u8(6);
                a.hash(state);
                b.hash(state);
            }
            TRecordConcat(a, b) => {
                state.write_u8(7);
                a.hash(state);
                b.hash(state);
            }
            TGeneric(a) => {
                state.write_u8(8);
                a.hash(state);
            }
            TArray(a, b) => {
                state.write_u8(9);
                a.hash(state);
                b.hash(state);
            }
            TArrayConcat(t0, t1) => {
                state.write_u8(10);
                t0.hash(state);
                t1.hash(state);
            }
            TUnit => {
                state.write_u8(11);
            }
            TNever => {
                state.write_u8(12);
            }
            TVar(a) => {
                state.write_u8(13);
                a.hash(state);
            }
            TError => {
                state.write_u8(14);
            }
        }
    }
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

impl Eq for Const {}
impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CInt(a), CInt(b)) => a == b,
            (CFloat(a), CFloat(b)) => a == b,
            (CBool(a), CBool(b)) => a == b,
            (CString(a), CString(b)) => a == b,
            (CUnit, CUnit) => true,
            (CChar(a), CChar(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub info: Info,
    pub kind: StmtKind,
}

pub use StmtKind::*;
#[derive(Clone, Debug)]
pub enum StmtKind {
    SDef(Meta, Name, Vector<Generic>, Vector<Pattern>, Type, Block),
    SRecDef(
        Meta,
        Vector<(Name, Vector<Generic>, Vector<Pattern>, Type, Block)>,
    ),
    SBif(Meta, Name, Vector<Generic>, Vector<Type>, Type),
    SEnum(Meta, Name, Vector<Generic>, Vector<(Name, Type)>),
    SBit(Meta, Name, Vector<Generic>),
    SVal(Pattern, Expr),
    SExpr(Expr),
    SNoop,
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub info: Info,
    pub t: Type,
    pub kind: Rc<ExprKind>,
}

pub use ExprKind::*;
#[derive(Clone, Debug)]
pub enum ExprKind {
    EConst(Const),
    EFun(Vector<Pattern>, Type, Block),
    EFunCall(Expr, Vector<Expr>),
    EFunCallDirect(Name, Vector<Type>, Vector<Expr>),
    EFunReturn(Expr),
    ELoop(Block),
    ELoopBreak(Expr),
    ELoopContinue,
    EMatch(Expr, Vector<Arm>),
    EArray(Vector<Expr>),
    EArrayConcat(Expr, Expr),
    EArrayAccess(Expr, Expr),
    EIfElse(Expr, Block, Block),
    ERecord(Vector<(Name, Expr)>),
    ERecordAccess(Expr, Name),
    ERecordConcat(Expr, Expr),
    EMut(Expr, Expr),
    EVal(Name),
    EVar(Name),
    EDef(Name, Vector<Type>),
    EVariant(Name, Vector<Type>, Name, Expr),
    EVariantAccess(Name, Vector<Type>, Name, Expr), // After pattern compilation
    EVariantCheck(Name, Vector<Type>, Name, Expr),
    EDo(Block),
    ENoop(Expr),
    ETuple(Vector<Expr>),
    ETupleAccess(Expr, Index),
    EFor(Pattern, Expr, Block),
    EWhile(Expr, Block),
    EError,
}

impl From<TypeKind> for Type {
    fn from(kind: TypeKind) -> Self {
        Type {
            kind: Rc::new(kind),
        }
    }
}

impl ExprKind {
    pub fn with(self, t: Type, info: Info) -> Expr {
        Expr {
            t,
            info,
            kind: Rc::new(self),
        }
    }
}

impl PatternKind {
    pub fn with(self, t: Type, info: Info) -> Pattern {
        Pattern {
            t,
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

impl Expr {
    pub fn kind(&self) -> ExprKind {
        (*self.kind).clone()
    }
    pub fn map_type(self, f: &impl Fn(Type) -> Type) -> Self {
        let t = f(self.t.clone());
        match self.kind() {
            EConst(l) => EConst(l),
            EMut(l, r) => EMut(l.map_type(f), r.map_type(f)),
            EDo(b) => EDo(b.map_type(f)),
            EFun(ps, t, b) => EFun(ps.map(|p| p.map_type(f)), f(t), b.map_type(f)),
            EFunCall(e, es) => {
                let e = e.map_type(f);
                let es = es.map(|e| e.map_type(f));
                EFunCall(e, es)
            }
            EFunCallDirect(x, ts, es) => {
                let ts = ts.map(f);
                let es = es.map(|e| e.map_type(f));
                EFunCallDirect(x, ts, es)
            }
            ELoop(b) => ELoop(b.map_type(f)),
            ELoopBreak(e) => ELoopBreak(e.map_type(f)),
            ELoopContinue => ELoopContinue,
            EMatch(e, arms) => {
                let e = e.map_type(f);
                let arms = arms.map(|(p, b)| (p.map_type(f), b.map_type(f)));
                EMatch(e, arms)
            }
            ERecord(xes) => {
                let xes = xes.map(|(x, e)| (x, e.map_type(f)));
                ERecord(xes)
            }
            ERecordAccess(e, x) => {
                let e = e.map_type(f);
                ERecordAccess(e, x)
            }
            EFunReturn(e) => {
                let e = e.map_type(f);
                EFunReturn(e)
            }
            EVal(x) => EVal(x),
            EVariant(xs, ts, x, e) => {
                let e = e.map_type(f);
                let ts = ts.map(f);
                EVariant(xs, ts, x, e)
            }
            ENoop(e) => {
                let e = e.map_type(f);
                ENoop(e)
            }
            EIfElse(e, b0, b1) => {
                let e = e.map_type(f);
                let b0 = b0.map_type(f);
                let b1 = b1.map_type(f);
                EIfElse(e, b0, b1)
            }
            EArray(es) => {
                let es = es.map(|e| e.map_type(f));
                EArray(es)
            }
            EArrayAccess(e0, e1) => {
                let e0 = e0.map_type(f);
                let e1 = e1.map_type(f);
                EArrayAccess(e0, e1)
            }
            EDef(x, ts) => EDef(x, ts.map(f)),
            EVar(x) => EVar(x),
            EError => EError,
            ETuple(es) => ETuple(es.map(|e| e.map_type(f))),
            ETupleAccess(e, i) => ETupleAccess(e.map_type(f), i),
            EFor(p, e, b) => EFor(p.map_type(f), e.map_type(f), b.map_type(f)),
            EWhile(e, b) => EWhile(e.map_type(f), b.map_type(f)),
            ERecordConcat(e0, e1) => ERecordConcat(e0.map_type(f), e1.map_type(f)),
            EVariantAccess(..) => unreachable!(),
            EVariantCheck(_, _, _, _) => unreachable!(),
            EArrayConcat(e0, e1) => EArrayConcat(e0.map_type(f), e1.map_type(f)),
        }
        .with(t, self.info)
    }
}

impl Block {
    pub fn map_type(mut self, f: &impl Fn(Type) -> Type) -> Self {
        self.ss = self.ss.map(|s| s.map_type(f));
        self.e = self.e.map_type(f);
        self
    }
}

impl Stmt {
    pub fn map_type(self, f: &impl Fn(Type) -> Type) -> Self {
        match self.kind {
            SVal(p, e) => {
                let p = p.map_type(f);
                let e = e.map_type(f);
                SVal(p, e)
            }
            SExpr(e) => {
                let e = e.map_type(f);
                SExpr(e)
            }
            SDef(m, x, gs, ps, t, b) => {
                let ps = ps.map(|p| p.map_type(f));
                let t = f(t);
                let b = b.map_type(f);
                SDef(m, x, gs, ps, t, b)
            }
            SBif(m, x, gs, ts, t) => {
                let ts = ts.map(f);
                let t = f(t);
                SBif(m, x, gs, ts, t)
            }
            SEnum(m, x, gs, xts) => {
                let xts = xts.map(|(x, t)| (x, f(t)));
                SEnum(m, x, gs, xts)
            }
            SBit(m, x, gs) => SBit(m, x, gs),
            SNoop => SNoop,
            SRecDef(m, ds) => todo!(),
        }
        .with(self.info)
    }
}

impl StmtKind {
    pub fn with(self, info: Info) -> Stmt {
        Stmt { info, kind: self }
    }
}

impl Pattern {
    pub fn kind(&self) -> PatternKind {
        (*self.kind).clone()
    }

    pub fn map_type(self, f: &impl Fn(Type) -> Type) -> Self {
        let t = f(self.t.clone());
        match self.kind() {
            PConst(l) => PConst(l),
            PVar(x) => PVar(x),
            PVal(x) => PVal(x),
            PRecord(xps) => {
                let xps = xps.map(|(x, p)| (x, p.map_type(f)));
                PRecord(xps)
            }
            PRecordConcat(p0, p1) => {
                let p0 = p0.map_type(f);
                let p1 = p1.map_type(f);
                PRecordConcat(p0, p1)
            }
            PTuple(ps) => {
                let ps = ps.map(|p| p.map_type(f));
                PTuple(ps)
            }
            PIgnore => PIgnore,
            PVariant(xs, ts, x, p) => {
                let p = p.map_type(f);
                let ts = ts.map(f);
                PVariant(xs, ts, x, p)
            }
            POr(p0, p1) => {
                let p0 = p0.map_type(f);
                let p1 = p1.map_type(f);
                POr(p0, p1)
            }
            PNoop(p) => {
                let p = p.map_type(f);
                PNoop(p)
            }
            PArray(ps) => {
                let ps = ps.map(|p| p.map_type(f));
                PArray(ps)
            }
            PArrayConcat(p0, p1) => {
                let p0 = p0.map_type(f);
                let p1 = p1.map_type(f);
                PArrayConcat(p0, p1)
            }
            PError => PError,
        }
        .with(t, self.info)
    }
}

pub fn row_to_fields(t: Type) -> Vector<(Name, Type)> {
    fn f(t: Type, xts: &mut Vector<(Name, Type)>) {
        match t.kind.as_ref().clone() {
            TRowEmpty => {}
            TRowExtend((x, t), r) => {
                xts.push_back((x, t));
                f(r, xts);
            }
            _ => unreachable!(),
        }
    }
    let mut xts = Vector::new();
    f(t, &mut xts);
    xts
}

pub fn fields_to_row(xts: Vector<(Name, Type)>) -> Type {
    xts.into_iter()
        .fold(TypeKind::TRowEmpty.into(), |r, (x, t)| {
            TypeKind::TRowExtend((x, t).into(), r).into()
        })
}
