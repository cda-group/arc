use crate::hir;
use crate::hir::Type;
use crate::info::files::Loc;

use arc_script_compiler_shared::Educe;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::MapEntry;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;

/// The `PlaceId` of a `Place`.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct PlaceId(pub(crate) usize);

#[derive(Debug, Copy, Clone, New, Educe)]
#[educe(PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Place {
    id: PlaceId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub(crate) t: Type,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(PartialOrd(ignore), Ord(ignore))]
    pub(crate) loc: Loc,
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
    pub(crate) loc: Loc,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct UseId(usize);

#[derive(Debug, Copy, Clone, New, Educe)]
#[educe(PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct Use {
    id: UseId,
    #[educe(PartialEq(ignore), Eq(ignore), Hash(ignore))]
    #[educe(Ord(ignore), PartialOrd(ignore))]
    pub(crate) loc: Loc,
}

#[derive(Debug, New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    #[shrinkwrap(main_field)]
    pub(crate) places: &'i mut PlaceInterner,
    pub(crate) hir: &'i hir::HIR,
}

#[derive(Default, Debug, New)]
pub(crate) struct PlaceInterner {
    place_counter: usize,
    use_counter: usize,
    branch_counter: usize,
    place_to_id: Map<PlaceKind, PlaceId>,
    pub(crate) roots: Vec<(hir::NameId, Place)>,
    pub(crate) parents: Vec<(Place, Place)>,
    pub(crate) uses: Vec<(Place, Use, Branch)>,
    pub(crate) jumps: Vec<(Branch, Branch)>,
}

impl PlaceInterner {
    pub(crate) fn intern_place(&mut self, place: PlaceKind) -> PlaceId {
        match self.place_to_id.entry(place) {
            MapEntry::Occupied(e) => *e.get(),
            MapEntry::Vacant(e) => {
                let id = PlaceId(self.place_counter);
                e.insert(id);
                self.place_counter += 1;
                id
            }
        }
    }
    pub(crate) fn new_branch(&mut self, loc: Loc) -> Branch {
        let b = Branch::new(BranchId(self.branch_counter), loc);
        self.branch_counter += 1;
        self.jumps.push((b, b));
        b
    }
    pub(crate) fn add_use(&mut self, p0: Place, b0: Branch, loc: Loc) {
        let id = UseId(self.use_counter);
        self.use_counter += 1;
        self.uses.push((p0, Use::new(id, loc), b0));
    }
    pub(crate) fn add_root(&mut self, p0: Place, x0: hir::Name) {
        self.roots.push((x0, p0))
    }
    pub(crate) fn add_parent(&mut self, p0: Place, p1: Place) {
        self.parents.push((p0, p1))
    }
    pub(crate) fn add_jump(&mut self, b0: Branch, b1: Branch) {
        self.jumps.push((b0, b1));
    }
}

impl hir::HIR {
    pub(crate) fn collect_places(&self) -> PlaceInterner {
        let mut places = PlaceInterner::default();
        let mut ctx = Context::new(&mut places, self);
        self.defs.values().for_each(|def| def.collect(&mut ctx));
        places
    }
}

impl hir::Block {
    fn collect_root(&self, ctx: &mut Context<'_>) {
        todo!()
    }
}

impl hir::Item {
    fn collect(&self, ctx: &mut Context<'_>) {
        match &self.kind {
            hir::ItemKind::Fun(i) => {
                i.body.collect_root(ctx);
            }
            hir::ItemKind::Task(i) => {
                for a in &i.assignments {
                    a.expr.collect_root(ctx);
                }
                if let Some(on) = &i.on {
                    on.body.collect_root(ctx);
                }
            }
            hir::ItemKind::Alias(_) => {}
            hir::ItemKind::Enum(_) => {}
            hir::ItemKind::Extern(_) => {}
            hir::ItemKind::Variant(_) => {}
        }
    }
}

impl hir::Expr {
    fn collect_root(&self, ctx: &mut Context<'_>) {
        let b = ctx.new_branch(self.loc);
        self.collect(b, ctx);
    }
    fn collect(&self, b0: Branch, ctx: &mut Context<'_>) -> Option<Place> {
        match ctx.hir.exprs.resolve(self) {
            /// Place-expressions
            hir::ExprKind::Access(e, x) => {
                if let Some(p0) = e.collect(b0, ctx) {
                    let p1 = Place::new(
                        ctx.intern_place(PlaceKind::Access(p0, *x)),
                        self.t,
                        self.loc,
                    );
                    ctx.add_parent(p0, p1);
                    return Some(p1);
                }
            }
            hir::ExprKind::Project(e, i) => {
                if let Some(p0) = e.collect(b0, ctx) {
                    let p1 = Place::new(
                        ctx.intern_place(PlaceKind::Project(p0, *i)),
                        self.t,
                        self.loc,
                    );
                    ctx.add_parent(p0, p1);
                    return Some(p1);
                }
            }
            hir::ExprKind::Var(x, kind) => match kind {
                hir::BindingKind::Member | hir::BindingKind::Local => {
                    let p = Place::new(ctx.intern_place(PlaceKind::Var(*x)), self.t, self.loc);
                    ctx.add_root(p, *x);
                    return Some(p);
                }
                hir::VarKind::State => {}
            },
            /// Branch/Use-expressions
            hir::ExprKind::If(e, b0, b1) => {
                todo!()
                //                 e0.collect_use(b0, owner);
                //                 let b1 = owner.new_branch(self.loc);
                //                 owner.add_jump(b0, b1);
                //                 e1.collect_use(b1, owner);
                //                 let b2 = owner.new_branch(self.loc);
                //                 owner.add_jump(b0, b2);
                //                 e2.collect_use(b2, owner);
            }
            hir::ExprKind::Loop(b) => {
                todo!()
                //                 let b1 = owner.new_branch(self.loc);
                //                 owner.add_jump(b, b1);
                //                 b.collect_use(b1, owner)
            }
            /// Use-expressions
            hir::ExprKind::BinOp(e0, _, e1) => {
                e0.collect_use(b0, ctx);
                e1.collect_use(b0, ctx);
            }
            hir::ExprKind::Call(e0, es) => {
                e0.collect_use(b0, ctx);
                es.iter().for_each(|e| e.collect_use(b0, ctx));
            }
            hir::ExprKind::Select(e0, es) => {
                e0.collect_use(b0, ctx);
                es.iter().for_each(|e| e.collect_use(b0, ctx));
            }
            hir::ExprKind::Struct(efs) => efs.values().for_each(|e| e.collect_use(b0, ctx)),
            hir::ExprKind::Array(es) => es.iter().for_each(|e| e.collect_use(b0, ctx)),
            hir::ExprKind::Tuple(es) => es.iter().for_each(|e| e.collect_use(b0, ctx)),
            hir::ExprKind::Emit(e0) => e0.collect_use(b0, ctx),
            hir::ExprKind::Log(e0) => e0.collect_use(b0, ctx),
            hir::ExprKind::UnOp(_, e0) => e0.collect_use(b0, ctx),
            hir::ExprKind::Enwrap(_, e0) => e0.collect_use(b0, ctx),
            hir::ExprKind::Unwrap(_, e0) => e0.collect_use(b0, ctx),
            hir::ExprKind::Item(_) => {}
            hir::ExprKind::Lit(_) => {}
            hir::ExprKind::Break => {}
            hir::ExprKind::After(_, _) => {}
            hir::ExprKind::Every(_, _) => {}
            hir::ExprKind::Cast(e, _) => e.collect_use(b0, ctx),
            hir::ExprKind::Unreachable => {}
            hir::ExprKind::Err => {}
            // NOTE: Ownership breaks for the following transformation:
            // if let Some(x) = Some(y) {
            //     ...
            // }
            // (Becomes)
            // if is[Some](y) {
            //     let x = unwrap[Some](y);
            //     ...
            // }
            // Therefore, `is` is a by-reference expression
            hir::ExprKind::Is(_, _) => {}
        }
        None
    }

    fn collect_use(&self, b0: Branch, ctx: &mut Context<'_>) {
        if let Some(p0) = self.collect(b0, ctx) {
            ctx.add_use(p0, b0, self.loc)
        }
    }
}
