use crate::compiler::hir;
use crate::compiler::info::files::Loc;
use crate::compiler::info::Info;
use crate::compiler::shared::New;

use std::collections::HashMap;

/// The PlaceId of a Place.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct PlaceId(pub(crate) usize);

#[derive(Debug, Copy, Clone, New, Educe)]
#[educe(PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Place {
    id: PlaceId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub(crate) loc: Option<Loc>,
}

/// A `Place` is an expression which refers to a memory location.
/// For example:
/// ```txt
/// a
/// b.0.1
/// c.d.5.e
/// ```
/// Two equivalent paths are guaranteed to have the same `PlaceId`.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub(crate) enum PlaceKind {
    Var(hir::Name),
    Project(PlaceId, hir::Index),
    Access(PlaceId, hir::Name),
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct BranchId(usize);

#[derive(Debug, Copy, Clone, New, Educe)]
#[educe(PartialEq, Eq, Hash)]
pub(crate) struct Branch {
    id: BranchId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    pub(crate) loc: Option<Loc>,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct UseId(usize);

#[derive(Debug, Copy, Clone, New, Educe)]
#[educe(PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct Use {
    id: UseId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(Ord(ignore), PartialOrd(ignore))]
    pub(crate) loc: Option<Loc>,
}

#[derive(Default, Debug)]
pub(crate) struct Ownership {
    place_counter: usize,
    use_counter: usize,
    branch_counter: usize,
    place_to_id: HashMap<PlaceKind, PlaceId>,
    pub(crate) roots: Vec<(Place, hir::NameId)>,
    pub(crate) parents: Vec<(Place, Place)>,
    pub(crate) uses: Vec<(Place, Use, Branch)>,
    pub(crate) jumps: Vec<(Branch, Branch)>,
}

use std::borrow::Borrow;
use std::collections::hash_map::Entry;

impl Ownership {
    pub(crate) fn intern_place(&mut self, place: PlaceKind) -> PlaceId {
        match self.place_to_id.entry(place) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let id = PlaceId(self.place_counter);
                e.insert(id);
                self.place_counter += 1;
                id
            }
        }
    }
    pub(crate) fn new_branch(&mut self, loc: Option<Loc>) -> Branch {
        let b = Branch::new(BranchId(self.branch_counter), loc);
        self.branch_counter += 1;
        self.jumps.push((b, b));
        b
    }
    pub(crate) fn add_use(&mut self, p0: Place, b0: Branch, loc: Option<Loc>) {
        let id = UseId(self.use_counter);
        self.use_counter += 1;
        self.uses.push((p0, Use::new(id, loc), b0));
    }
    pub(crate) fn add_root(&mut self, p0: Place, x0: hir::Name) {
        self.roots.push((p0, x0.id))
    }
    pub(crate) fn add_parent(&mut self, p0: Place, p1: Place) {
        self.parents.push((p0, p1))
    }
    pub(crate) fn add_jump(&mut self, b0: Branch, b1: Branch) {
        self.jumps.push((b0, b1));
    }
}

impl From<&'_ hir::HIR> for Ownership {
    fn from(hir: &hir::HIR) -> Self {
        let mut owner = Ownership::default();
        hir.defs.values().for_each(|def| def.collect(&mut owner));
        owner
    }
}

impl hir::Item {
    fn collect(&self, places: &mut Ownership) {
        match &self.kind {
            hir::ItemKind::Alias(_) => {}
            hir::ItemKind::Enum(_) => {}
            hir::ItemKind::Fun(i) => {
                i.body.collect_root(places);
            }
            hir::ItemKind::State(i) => {
                i.init.collect_root(places);
            }
            hir::ItemKind::Task(i) => {
                i.on.body.collect_root(places);
            }
            hir::ItemKind::Extern(_) => {}
        }
    }
}

impl hir::Expr {
    fn collect_root(&self, owner: &mut Ownership) {
        let b = owner.new_branch(self.loc);
        self.collect(b, owner);
    }
    fn collect(&self, b0: Branch, owner: &mut Ownership) -> Option<Place> {
        match &self.kind {
            /// Place-expressions
            hir::ExprKind::Access(e, x) => {
                if let Some(p0) = e.collect(b0, owner) {
                    let p1 = Place::new(owner.intern_place(PlaceKind::Access(p0.id, *x)), self.loc);
                    owner.add_parent(p0, p1);
                    return Some(p1);
                }
            }
            hir::ExprKind::Project(e, i) => {
                if let Some(p0) = e.collect(b0, owner) {
                    let p1 =
                        Place::new(owner.intern_place(PlaceKind::Project(p0.id, *i)), self.loc);
                    owner.add_parent(p0, p1);
                    return Some(p1);
                }
            }
            hir::ExprKind::Var(x) => {
                let p = Place::new(owner.intern_place(PlaceKind::Var(*x)), self.loc);
                owner.add_root(p, *x);
                return Some(p);
            }
            /// Branch/Use-expressions
            hir::ExprKind::If(e0, e1, e2) => {
                e0.collect_use(b0, owner);
                let b1 = owner.new_branch(self.loc);
                owner.add_jump(b0, b1);
                e1.collect_use(b1, owner);
                let b2 = owner.new_branch(self.loc);
                owner.add_jump(b0, b2);
                e2.collect_use(b2, owner);
            }
            hir::ExprKind::Loop(e0) => {
                let b1 = owner.new_branch(self.loc);
                owner.add_jump(b0, b1);
                e0.collect_use(b1, owner)
            }
            /// Use-expressions
            hir::ExprKind::BinOp(e0, _, e1) => {
                e0.collect_use(b0, owner);
                e1.collect_use(b0, owner);
            }
            hir::ExprKind::Call(e0, es) => {
                e0.collect_use(b0, owner);
                es.iter().for_each(|e| e.collect_use(b0, owner));
            }
            hir::ExprKind::Let(_, e0, e1) => {
                e0.collect_use(b0, owner);
                e1.collect_use(b0, owner);
            }
            hir::ExprKind::Struct(efs) => efs.values().for_each(|e| e.collect_use(b0, owner)),
            hir::ExprKind::Array(es) => es.iter().for_each(|e| e.collect_use(b0, owner)),
            hir::ExprKind::Tuple(es) => es.iter().for_each(|e| e.collect_use(b0, owner)),
            hir::ExprKind::Emit(e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Log(e0) => e0.collect_use(b0, owner),
            hir::ExprKind::UnOp(_, e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Enwrap(_, _, e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Unwrap(_, e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Is(_, e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Return(e0) => e0.collect_use(b0, owner),
            hir::ExprKind::Item(_) => {}
            hir::ExprKind::Lit(_) => {}
            hir::ExprKind::Break => {}
            hir::ExprKind::Err => {}
        }
        None
    }

    fn collect_use(&self, b0: Branch, owner: &mut Ownership) {
        if let Some(p0) = self.collect(b0, owner) {
            owner.add_use(p0, b0, self.loc)
        }
    }
}
