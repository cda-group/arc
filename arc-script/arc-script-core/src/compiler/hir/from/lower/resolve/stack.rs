use crate::compiler::hir::Name;
use crate::compiler::info::diags::{Error, Result};
use crate::compiler::info::Info;

use shrinkwraprs::Shrinkwrap;

use std::collections::hash_map::Entry;
use std::collections::HashMap as Map;

/// A scope which maps variable names to unique variable names.
#[derive(Debug, Default, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Scope(pub(crate) Map<Name, Name>);

#[derive(Debug, Default, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Frame(pub(crate) Vec<Scope>);

/// A data structure which stores information about variables and scopes.
/// Note that the `SymbolStack` does not store information about items and
/// namespaces. This information is stored in the `SymbolTable`.
///
/// The difference between a variable and item is that multiple variables
/// with the same name can live in the same namespace, as long as they
/// reside in different scopes. Items on the other hand must live in
/// different namespaces. This restriction allows items to be referenced
/// through paths.
///
/// Because multiple variables can exist in the same namespace, variables
/// may not be referenced through paths as this would lead to ambiguity.
///
/// It is the job of the `SymbolStack` to eliminate this ambiguity by
/// generating a unique identifier for each variable which is the the
/// variable name + a the current value of a counter. Uniqueness is
/// important for code-generation to work. The variable name is unique
/// within the function, but not across multiple monomorphised instances
/// of the function. For this reason, the identifier is not stored in the
/// `SymbolTable`.
///
/// It is however possible to make the assumption that all variables are
/// declared within the scope of a function or event handler. To this end,
/// functions and event handlers store a `VariableTable` for quick lookup
/// of identifiers during type inference. Another possible alternative is
/// to again construct a stack during type inference.
///
/// Design questions to consider:
/// * Should variables be able to shadow items and/or other variables?
/// * Should closures be able to capture the environment?
///
#[derive(Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct SymbolStack(pub(crate) Vec<Frame>);

impl Default for SymbolStack {
    fn default() -> Self {
        Self(vec![Frame::default()])
    }
}

impl SymbolStack {
    /// Returns the true name of a symbol.
    pub(crate) fn resolve(&self, name: Name) -> Option<Name> {
        self.innermost()
            .iter()
            .rev()
            .find_map(|scope| scope.get(&name))
            .cloned()
    }

    /// Returns the active frame.
    fn innermost(&self) -> &Vec<Scope> {
        self.iter().last().unwrap()
    }

    /// Returns the active frame as mutable.
    fn innermost_mut(&mut self) -> &mut Vec<Scope> {
        self.iter_mut().last().unwrap()
    }

    /// Binds a local variable to a name in the innermost scope. Returns an error
    /// if there is already a variable in that scope with the same name. This might
    /// occur if for example two variables are bound in the same pattern.
    pub(crate) fn bind(&mut self, name: Name, info: &mut Info) -> Option<Name> {
        match self.innermost_mut().last_mut().unwrap().entry(name) {
            Entry::Vacant(entry) => {
                let uid = info.names.fresh_with_base(name);
                entry.insert(uid);
                Some(uid)
            }
            Entry::Occupied(_) => None,
        }
    }

    /// Pushes a new local scope onto the last frame of the stack.
    pub(crate) fn push_scope(&mut self) {
        self.innermost_mut().push(Scope::default())
    }

    /// Pops a local scope from the last frame of the stack.
    pub(crate) fn pop_scope(&mut self) {
        self.innermost_mut().pop();
    }

    /// Pushes a new frame and scope onto the stack.
    pub(crate) fn push_frame(&mut self) {
        self.push(Frame::default());
        self.push_scope();
    }

    pub(crate) fn pop_frame(&mut self) {
        self.pop();
    }
}
