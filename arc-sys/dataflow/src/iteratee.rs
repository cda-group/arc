pub trait Iteratee {
    type Item;
    fn feed(&mut self, input: Self::Item);
}

impl<T> Iteratee for &mut T
where
    T: Iteratee,
{
    type Item = T::Item;
    fn feed(&mut self, input: Self::Item) {
        T::feed(self, input)
    }
}
