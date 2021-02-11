use crate::compiler::hir;
use crate::compiler::hir::from::check::ownership::relations::Branch;
use crate::compiler::hir::from::check::ownership::relations::Ownership;
use crate::compiler::hir::from::check::ownership::relations::Place;
use crate::compiler::hir::from::check::ownership::relations::Use;
use crate::compiler::info::diags::Error;
use crate::compiler::info::Info;
use crepe::crepe;

crepe! {

    @input
    // NameId is the direct root of Place, e.g. Root(a, a.b)
    #[derive(Debug)]
    pub(crate) struct Root(hir::NameId, Place);

    @input
    // Place is a direct parent of Place, e.g. Parent(a.b, a.b.c)
    #[derive(Debug)]
    pub(crate) struct Parent(Place, Place);

    // Place is used in Use of Branch, e.g. Used(a.b, <id>, ..) where let x = <id> in e
    @input
    #[derive(Debug)]
    pub(crate) struct Used(Place, Use, Branch);

    // Branch has a direct jump to Branch, e.g. Jump(if x { 1 } else { 2 }, { 1 })
    @input
    #[derive(Debug)]
    pub(crate) struct Jump(Branch, Branch);

    // NameId is the transitive root of Place, e.g. Origin(a, a.b.c)
    struct Origin(hir::NameId, Place);

    // Place is a transitive ancestor of Place, e.g. Ancestor(a.b, a.b.c.d)
    struct Ancestor(Place, Place);

    // Branch can be transitively reached from Branch,
    // e.g. Jump(if x { if y { 1 } else { 2 } } else { 3 }, { 1 })
    struct Reachable(Branch, Branch);

    // Origin: Direct-case
    Origin(x, p0) <-
        Root(x, p0);

    // Origin: Transitive case
    Origin(x, p1) <-
        Root(x, p0),
        Ancestor(p0, p1);

    // Ancestor: Direct case
    Ancestor(p0, p1) <-
        Parent(p0, p1);

    // Ancestor: Transitive case
    Ancestor(p0, p2) <-
        Parent(p0, p1),
        Ancestor(p1, p2);

    // Reachable: Direct-case
    Reachable(b0, b1) <-
        Jump(b0, b1);

    // Reachable: Transitive-case
    Reachable(b0, b2) <-
        Jump(b0, b1),
        Reachable(b1, b2);

    // Reachable: Reflexive-case
    Reachable(b0, b0) <-
        Jump(b0, _);

    // Error when Place is an Ancestor of Place and is used in Use.
    @output
    #[derive(Debug, PartialOrd, Ord)]
    pub(crate) struct UseOfMovedValue(Place, Place, Use);

    // Error when Place is used in both Use and Use.
    @output
    #[derive(Debug, PartialOrd, Ord)]
    pub(crate) struct DoubleUse(Place, Use, Use);

    // An ancestor of a child cannot be used.
    UseOfMovedValue(p0, p1, u0) <-
        Used(p0, u0, b0),
        Used(p1, _, b1),
        Ancestor(p0, p1),
        Reachable(b0, b1);

    // A Place cannot be used twice.
    DoubleUse(p0, u0, u1) <-
        Used(p0, u0, b0),
        Used(p0, u1, b1),
        Reachable(b0, b1),
        (u0 < u1);
}

use std::collections::HashSet;

impl hir::HIR {
    pub(crate) fn check_ownership(&self, info: &mut Info) {
        let mut places = Ownership::from(self);
        let mut runtime = Crepe::new();

        runtime.extend(places.roots.drain(..).map(|(x0, p0)| Root(p0, x0)));
        runtime.extend(places.uses.drain(..).map(|(p0, u0, b0)| Used(p0, u0, b0)));
        runtime.extend(places.parents.drain(..).map(|(p0, p1)| Parent(p0, p1)));
        runtime.extend(places.jumps.drain(..).map(|(b0, b1)| Jump(b0, b1)));

        let (errs0, errs1) = runtime.run();

        let mut errs0 = errs0.into_iter().collect::<Vec<_>>();
        let mut errs1 = errs1.into_iter().collect::<Vec<_>>();

        errs0.sort();
        errs1.sort();

        for UseOfMovedValue(p0, p1, u0) in errs0 {
            info.diags.intern(Error::UseOfMovedValue {
                loc0: p0.loc,
                loc1: p1.loc,
            })
        }

        for DoubleUse(p0, u0, u1) in errs1 {
            info.diags.intern(Error::DoubleUse {
                loc0: p0.loc,
                loc1: u0.loc,
                loc2: u1.loc,
            })
        }
    }
}
