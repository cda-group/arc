use crate::{prelude::*, codegen::printer::Printer};
use std::fmt::{self, Display, Formatter};

/// Wraps a single AST node to be printed
#[derive(Constructor)]
pub struct Pretty<'i, T>(pub &'i T, pub &'i Printer<'i>);

pub trait ToPretty: Sized {
    fn pretty<'i>(&'i self, pr: &'i Printer) -> Pretty<'i, Self> {
        Pretty::new(self, pr)
    }
}

impl<T> ToPretty for T {}

/// Wraps a list of AST nodes to be printed
#[derive(Constructor)]
pub struct AllPretty<'i, I> {
    pub iter: I,
    pub sep: &'i str,
    pub pr: &'i Printer<'i>,
}

pub trait ToAllPretty: Sized {
    fn all_pretty<'i, T>(self, sep: &'i str, pr: &'i Printer<'i>) -> AllPretty<'i, Self>
    where
        Self: IntoIterator<Item = &'i T> + Clone,
        T: 'i,
    {
        AllPretty::new(self, sep, pr)
    }
}

impl<T> ToAllPretty for T {}

impl<'i, I, T> Display for AllPretty<'i, I>
where
    I: IntoIterator<Item = &'i T> + Clone,
    for<'x> Pretty<'x, T>: Display,
    T: 'i,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.clone().into_iter();
        if let Some(item) = iter.next() {
            write!(f, "{}", item.pretty(&self.pr))?;
        }
        for item in iter {
            write!(f, "{}{}", self.sep, item.pretty(&self.pr))?;
        }
        Ok(())
    }
}

/// Wraps a list of functions to be printed
#[derive(Constructor)]
pub struct MapPretty<'i, I, F> {
    pub iter: I,
    pub mapper: F,
    pub sep: &'i str,
}

pub trait ToMapPretty: Sized {
    fn map_pretty<'i, F, T>(self, f: F, sep: &'i str) -> MapPretty<'i, Self, F>
    where
        Self: IntoIterator<Item = &'i T> + Clone,
        for<'r, 's> F: Fn(&'i T, &'r mut Formatter<'s>) -> fmt::Result,
        T: 'i,
    {
        MapPretty::new(self, f, sep)
    }
}

impl<T> ToMapPretty for T {}

/// Lazily prints an iterator of displayable items.
/// Feels like this could be useful to have in Rust's standard library.
impl<'i, I, T, F> Display for MapPretty<'i, I, F>
where
    I: IntoIterator<Item = &'i T> + Clone,
    for<'r, 's> F: Fn(&'i T, &'r mut Formatter<'s>) -> fmt::Result,
    T: 'i,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut iter = self.iter.clone().into_iter();
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
