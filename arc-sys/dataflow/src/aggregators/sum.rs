use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div};

use num::NumCast;
use num::Zero;

use crate::aggregator::Aggregator;
use crate::data::Data;

struct Sum<T>(PhantomData<T>);

impl<T> Aggregator for Sum<T>
where
    T: Data + Zero,
{
    type I = T;
    type P = T;
    type O = T;

    fn lift(input: Self::I) -> Self::P {
        input
    }

    fn merge(a: Self::P, b: Self::P) -> Self::P {
        a + b
    }

    fn identity() -> Self::P {
        Self::P::zero()
    }

    fn lower(output: Self::P) -> Self::O {
        output
    }
}

struct Count<T>(PhantomData<T>);

impl<T> Aggregator for Count<T>
where
    T: Data,
{
    type I = T;
    type P = u64;
    type O = u64;

    fn lift(input: Self::I) -> Self::P {
        1
    }

    fn merge(a: Self::P, b: Self::P) -> Self::P {
        a + b
    }

    fn identity() -> Self::P {
        0
    }

    fn lower(output: Self::P) -> Self::O {
        output
    }
}

struct Average<T>(PhantomData<T>);

impl<T> Aggregator for Average<T>
where
    T: Data + Zero + Div<Output = T> + NumCast,
{
    type I = T;
    type P = (<Sum<T> as Aggregator>::P, <Count<T> as Aggregator>::P);
    type O = T;

    fn lift(input: Self::I) -> Self::P {
        (Sum::<T>::lift(input.clone()), Count::<T>::lift(input))
    }

    fn merge(a: Self::P, b: Self::P) -> Self::P {
        (Sum::<T>::merge(a.0, b.0), Count::<T>::merge(a.1, b.1))
    }

    fn identity() -> Self::P {
        (Sum::<T>::identity(), Count::<T>::identity())
    }

    fn lower(output: Self::P) -> Self::O {
        Sum::<T>::lower(output.0) / NumCast::from(Count::<T>::lower(output.1)).unwrap()
    }
}
