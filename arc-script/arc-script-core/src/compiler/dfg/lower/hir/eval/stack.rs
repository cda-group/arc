use crate::compiler::dfg::lower::hir::eval::value::Value;
use crate::compiler::hir::Path;
use crate::compiler::info::names::NameId;

use arc_script_core_shared::Map;
use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;

/// A stack frame of variables and their values. Corresponds to the scope
/// of a function.
/// NB: Closures are functions after lambda lifting.
#[derive(New, Debug, Shrinkwrap, Clone)]
#[shrinkwrap(mutable)]
pub(crate) struct Frame {
    pub(crate) path: Path,
    #[shrinkwrap(main_field)]
    pub(crate) data: Map<NameId, Value>,
}

/// A stack for storing frames. Each frame stores a map of variables.
/// Only variables in the outermost frame may be accessed by the interpreter.
/// This is similar to the symbol-stack in the resolver, but assumes that all
/// variable names are unique, which means scopes are not necessary.
#[derive(Default, Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Stack(pub(crate) Vec<Frame>);

impl Stack {
    /// Pushes a frame onto the stack.
    pub(crate) fn push_frame(&mut self, path: Path) {
        self.push(Frame::new(path, Map::default()));
    }
    /// Pops a frame offof the stack.
    pub(crate) fn pop_frame(&mut self) {
        self.pop();
    }
    pub(crate) fn take_frame(&mut self) -> Frame {
        self.pop().unwrap()
    }
    /// Inserts a variable in the outermost frame of the stack.
    ///
    /// TODO: Probably it is better to pre-allocate the stack-frames instead
    /// of inserting variables as they are encountered.
    pub(crate) fn insert(&mut self, k: NameId, v: Value) {
        self.last_mut().unwrap().insert(k, v);
    }
    /// Returns the value of a variable in the outermost frame of the stack.
    /// NB: Only programs which are well-typed are evaluated. For this reason,
    /// unwrapping is OK since well-typedness implies names have resolved correctly.
    pub(crate) fn lookup(&self, k: NameId) -> &Value {
        self.last().and_then(|frame| frame.get(&k)).unwrap()
    }
    /// Mutates a variable in the outermost frame of the stack.
    pub(crate) fn update(&mut self, k: NameId, v: Value) {
        let res = self.last_mut().and_then(|frame| frame.insert(k, v));
        debug_assert!(res.is_some())
    }
}
