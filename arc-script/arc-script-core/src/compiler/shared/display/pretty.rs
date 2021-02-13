use crate::compiler::shared::display::format::Context;
use crate::compiler::shared::New;

use std::cell::Cell;
use std::fmt::{self, Display, Formatter};

/// Wraps a generic node to be printed in a specific context.
/// NB: This is a tuple struct since it makes the syntax more concise.
#[derive(New)]
pub(crate) struct Pretty<'i, Node, State: Copy>(pub(crate) &'i Node, pub(crate) Context<State>);

/// Creates a pretty format for an AST node.
pub(crate) trait AsPretty: Sized {
    /// Wraps `Self` and context `T` inside a struct which can be pretty-printed.
    fn pretty<State, T>(&self, ctx: T) -> Pretty<'_, Self, State>
    where
        T: AsRef<Context<State>>,
        State: Copy,
    {
        Pretty::new(self, *ctx.as_ref())
    }

    /// Wraps `Self` inside a struct which can be pretty-printed. This is similar to [`pretty`] but
    /// initializes the pretty-printer context (i.e., gives it an initial indentation).
    fn to_pretty<State>(&self, state: State) -> Pretty<'_, Self, State>
    where
        State: Copy,
    {
        Pretty::new(self, Context::with_state(state))
    }
}

/// Any AST node can be pretty printed as long as it implements `Display` for `Pretty<Node>`.
impl<Node> AsPretty for Node {}

/// Wraps a list of AST nodes to be pretty printed.
#[derive(New)]
pub(crate) struct AllPretty<'i, Iter, State>
where
    State: Copy,
{
    pub(crate) iter: Cell<Option<Iter>>,
    pub(crate) sep: &'i str,
    pub(crate) ctx: Context<State>,
}

/// Pretty prints an iterator of AST nodes using a separator.
pub(crate) trait ToAllPretty: Sized {
    fn all_pretty<Node, State, T>(self, sep: &str, ctx: T) -> AllPretty<'_, Self, State>
    where
        Self: IntoIterator<Item = Node>,
        T: AsRef<Context<State>>,
        State: Copy,
    {
        AllPretty::new(Cell::new(Some(self)), sep, *ctx.as_ref())
    }
}

impl<Node> ToAllPretty for Node {}

impl<'i, Iter, Node, State> Display for AllPretty<'i, Iter, State>
where
    Iter: IntoIterator<Item = &'i Node>,
    for<'x> Pretty<'x, Node, State>: Display,
    Node: 'i,
    State: Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.take().unwrap().into_iter();
        if let Some(item) = iter.next() {
            write!(f, "{}", item.pretty(self.ctx))?;
        }
        for item in iter {
            write!(f, "{}{}", self.sep, item.pretty(&self.ctx))?;
        }
        Ok(())
    }
}

/// Wraps a list of functions to be printed.
#[derive(New)]
pub(crate) struct MapPretty<'i, Iter, Fun> {
    pub(crate) iter: Cell<Option<Iter>>,
    pub(crate) mapper: Fun,
    pub(crate) sep: &'i str,
}

/// Pretty prints a list of nodes using a mapper function.
pub(crate) trait ToMapPretty {
    fn map_pretty<Fun, Node>(self, fun: Fun, sep: &str) -> MapPretty<'_, Self, Fun>
    where
        Self: IntoIterator<Item = Node> + Sized,
        for<'r, 's> Fun: Fn(Node, &'r mut Formatter<'s>) -> fmt::Result,
    {
        MapPretty::new(Cell::new(Some(self)), fun, sep)
    }
}

impl<Node> ToMapPretty for Node {}

/// Lazily prints an iterator of displayable items.
/// Feels like this could be useful to have in Rust's standard library.
impl<'i, Iter, Node, Fun> Display for MapPretty<'i, Iter, Fun>
where
    Iter: IntoIterator<Item = Node>,
    for<'r, 's> Fun: Fn(Node, &'r mut Formatter<'s>) -> fmt::Result,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.take().unwrap().into_iter();
        if let Some(item) = iter.next() {
            (self.mapper)(item, f)?;
        }
        for item in iter {
            write!(f, "{}", self.sep)?;
            (self.mapper)(item, f)?;
        }
        Ok(())
    }
}
