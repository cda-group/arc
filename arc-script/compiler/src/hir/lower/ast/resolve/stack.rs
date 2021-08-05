use crate::hir;
use crate::info::Info;

use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::MapEntry;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;

/// A data structure which stores information about locals and scopes.
/// Note that the `SymbolStack` does not store information about items and
/// namespaces. This information is stored in the `SymbolTable`.
///
/// The difference between a local and an item is that multiple locals
/// with the same name can live in the same namespace, as long as they
/// reside in different scopes. Items on the other hand must live in
/// different namespaces. This restriction allows items to be referenced
/// through paths.
///
/// Because multiple locals can exist in the same namespace, locals
/// may not be referenced through paths as this would lead to ambiguity.
///
/// It is the job of the `SymbolStack` to eliminate this ambiguity by
/// generating a unique identifier for each local which is the the
/// local name + the current value of a counter. Uniqueness is
/// important for code-generation to work. The local's name is unique
/// within the function, but not across multiple monomorphised instances
/// of the function. For this reason, the identifier is not stored in the
/// `SymbolTable`.
///
/// Design questions to consider:
/// * Should variables be able to shadow items and/or other variables?
/// * Should closures be able to capture the environment?
///
#[derive(Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct SymbolStack(pub(crate) Vec<Frame>);

/// A frame of a function which stores a stack of scopes. Frames are mainly used to deal with
/// lambdas. Lambdas can for the moment not capture variables, therefore each lambda has its
/// own frame.
#[derive(Debug, Default, Clone, Shrinkwrap, New)]
#[shrinkwrap(mutable)]
pub(crate) struct Frame(pub(crate) Vec<Scope>);

/// A scope which maps variable names to unique variable names.
#[derive(Debug, Default, Clone, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Scope {
    #[shrinkwrap(main_field)]
    pub(crate) vars: Map<hir::Name, (hir::Name, hir::ScopeKind)>,
    pub(crate) stmts: VecDeque<hir::Stmt>,
}

impl Default for SymbolStack {
    fn default() -> Self {
        Self(vec![Frame::default()])
    }
}

impl SymbolStack {
    /// Returns the true name of a symbol.
    pub(crate) fn resolve(&self, name: hir::Name) -> Option<(hir::Name, hir::ScopeKind)> {
        self.innermost()
            .iter()
            .rev()
            .find_map(|scope| scope.get(&name))
            .copied()
    }

    /// Returns the active frame.
    fn innermost(&self) -> &Vec<Scope> {
        self.iter().last().unwrap()
    }

    /// Returns the active frame as mutable.
    fn innermost_mut(&mut self) -> &mut Vec<Scope> {
        self.iter_mut().last().unwrap()
    }

    pub(crate) fn rename_to_unique(
        &mut self,
        name: hir::Name,
        kind: hir::ScopeKind,
        info: &mut Info,
    ) -> Option<hir::Name> {
        self.rename(name, || info.names.fresh_with_base(name), kind)
    }

    /// Binds a local variable to a name in the innermost scope. Returns `None`
    /// if there is already a variable in that scope with the same name. This might
    /// occur if for example two variables are bound in the same pattern.
    pub(crate) fn rename(
        &mut self,
        name: hir::Name,
        mut uid: impl FnMut() -> hir::Name,
        kind: hir::ScopeKind,
    ) -> Option<hir::Name> {
        match self.innermost_mut().last_mut().unwrap().entry(name) {
            MapEntry::Vacant(entry) => {
                let uid = uid();
                entry.insert((uid, kind));
                Some(uid)
            }
            MapEntry::Occupied(_) => None,
        }
    }

    /// Pushes a new local scope onto the last frame of the stack.
    pub(crate) fn push_scope(&mut self) {
        self.innermost_mut().push(Scope::default())
    }

    /// Pops a local scope from the last frame of the stack.
    #[must_use]
    pub(crate) fn pop_scope(&mut self) -> VecDeque<hir::Stmt> {
        self.innermost_mut().pop().unwrap().stmts
    }

    /// Pushes a new frame and scope onto the stack.
    pub(crate) fn push_frame(&mut self) {
        self.push(Frame::default());
        self.push_scope();
    }

    /// Pops the innermost frame off of the stack.
    #[must_use]
    pub(crate) fn pop_frame(&mut self) -> VecDeque<hir::Stmt> {
        let stmts = self.pop_scope();
        self.pop();
        stmts
    }

    pub(crate) fn stmts(&mut self) -> &mut VecDeque<hir::Stmt> {
        &mut self.innermost_mut().last_mut().unwrap().stmts
    }
}
