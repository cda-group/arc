//! This module allows pretty printers to be implemented for language constructs.

use arc_script_compiler_shared::Format;
use arc_script_compiler_shared::New;

use std::cell::Cell;
use std::fmt::{self, Display, Formatter};

// Macro for implementing pretty printers
macro_rules! pretty {
    {
        [$node:ident, $fmt:ident, $w:ident]
        $($path:path => $expr:expr ,)*
    } => {
        $(
            impl<'i> Display for Pretty<'i, $path, Context<'_>> {
                fn fmt(&self, w: &mut Formatter<'_>) -> fmt::Result {
                    let $w = w;
                    let $node = self.node;
                    let $fmt = self.fmt;
                    $expr;
                    Ok(())
                }
            }
        )*
    };
}

/// Wraps a generic node to be printed in a specific context.
#[derive(New)]
pub(crate) struct Pretty<'i, Node, Context: Copy> {
    pub(crate) node: &'i Node,
    pub(crate) fmt: Format<Context>,
}

/// Creates a pretty format for an AST node.
pub(crate) trait AsPretty: Sized {
    /// Wraps `Self` and context `T` inside a struct which can be pretty-printed.
    fn pretty<Context, T>(&self, fmt: T) -> Pretty<'_, Self, Context>
    where
        T: AsRef<Format<Context>>,
        Context: Copy,
    {
        Pretty::new(self, *fmt.as_ref())
    }

    /// Wraps `Self` inside a struct which can be pretty-printed. This is similar to [`pretty`] but
    /// initializes the pretty-printer context (i.e., gives it an initial indentation).
    fn to_pretty<Context>(&self, ctx: Context) -> Pretty<'_, Self, Context>
    where
        Context: Copy,
    {
        Pretty::new(self, Format::with_ctx(ctx))
    }
}

/// Any AST node can be pretty printed as long as it implements `Display` for `Pretty<Node>`.
impl<Node> AsPretty for Node {}

/// Wraps a list of AST nodes to be pretty printed.
#[derive(New)]
pub(crate) struct AllPretty<'i, Iter, Context>
where
    Context: Copy,
{
    pub(crate) iter: Cell<Option<Iter>>,
    pub(crate) sep: &'i str,
    pub(crate) ctx: Format<Context>,
}

/// Pretty prints an iterator of AST nodes using a separator.
pub(crate) trait ToAllPretty: Sized {
    fn all_pretty<Node, Context, T>(self, sep: &str, fmt: T) -> AllPretty<'_, Self, Context>
    where
        Self: IntoIterator<Item = Node>,
        T: AsRef<Format<Context>>,
        Context: Copy,
    {
        AllPretty::new(Cell::new(Some(self)), sep, *fmt.as_ref())
    }
}

impl<Node> ToAllPretty for Node {}

impl<'i, Iter, Node, Context> Display for AllPretty<'i, Iter, Context>
where
    Iter: IntoIterator<Item = &'i Node>,
    for<'x> Pretty<'x, Node, Context>: Display,
    Node: 'i,
    Context: Copy,
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
