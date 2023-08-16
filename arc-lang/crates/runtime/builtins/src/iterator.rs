use crate::cow::Cow;
use crate::traits::Data;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Iter<T, I: std::iter::Iterator<Item = T> + Clone>(Cow<I>);

impl<T, I: std::iter::Iterator<Item = T> + Clone> Iter<T, I> {
    pub fn new(i: I) -> Iter<T, I> {
        Iter(Cow::new(i))
    }

    pub fn enumerate(self) -> Iter<(usize, T), std::iter::Enumerate<I>> {
        Iter(Cow::new(self.0.take().enumerate()))
    }

    pub fn map<U, F: FnMut(T) -> U + Clone>(self, f: F) -> Iter<U, std::iter::Map<I, F>> {
        Iter(Cow::new(self.0.take().map(f)))
    }

    pub fn filter<F: FnMut(&T) -> bool + Clone>(self, f: F) -> Iter<T, std::iter::Filter<I, F>> {
        Iter(Cow::new(self.0.take().filter(f)))
    }
}

impl<T: Data, I: std::iter::Iterator<Item = T> + Clone> Iterator for Iter<T, I> {
    type Item = T;

    fn next(&mut self) -> std::option::Option<T> {
        self.0.update(|this| match this.next() {
            Some(x) => std::option::Option::Some(x),
            None => std::option::Option::None,
        })
    }
}
