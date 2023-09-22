use im_rc::OrdMap;
use im_rc::Vector;

pub trait VectorUtils<T0> {
    fn mapm<T1, C>(self, ctx: &mut C, f: impl FnMut(&mut C, T0) -> T1) -> Vector<T1>
    where
        T1: Clone;

    fn zip<T1>(&self, other: &Vector<T1>) -> Vector<(T0, T1)>
    where
        T1: Clone;

    fn filter_mapm<T1, C>(self, ctx: &mut C, f: impl FnMut(&mut C, T0) -> Option<T1>) -> Vector<T1>
    where
        T1: Clone;

    fn map<T1>(self, f: impl FnMut(T0) -> T1) -> Vector<T1>
    where
        T1: Clone;

    fn concat(self, other: Vector<T0>) -> Vector<T0>;
}

pub trait AssocVectorUtils<K, T0> {
    fn mapm_assoc<T1, C>(self, ctx: &mut C, f: impl FnMut(&mut C, T0) -> T1) -> Vector<(K, T1)>
    where
        K: Clone,
        T1: Clone;

    fn find_assoc(&self, k: &K) -> Option<&T0>
    where
        K: Clone + PartialEq,
        T0: Clone;

    fn remove_assoc(&mut self, k: &K) -> Option<T0>
    where
        K: Clone + PartialEq,
        T0: Clone;
}

impl<K, T0> AssocVectorUtils<K, T0> for Vector<(K, T0)>
where
    K: Clone,
    T0: Clone,
{
    fn mapm_assoc<T1, C>(self, ctx: &mut C, mut f: impl FnMut(&mut C, T0) -> T1) -> Vector<(K, T1)>
    where
        K: Clone,
        T1: Clone,
    {
        self.map(|(k, v)| (k, f(ctx, v)))
    }

    fn find_assoc(&self, k: &K) -> Option<&T0>
    where
        K: Clone + PartialEq,
        T0: Clone,
    {
        self.iter()
            .find_map(|(k1, v)| if k1 == k { Some(v) } else { None })
    }

    fn remove_assoc(&mut self, k: &K) -> Option<T0>
    where
        K: Clone + PartialEq,
        T0: Clone,
    {
        for (i, (k1, _)) in self.iter().enumerate() {
            if k1 == k {
                let (_, v) = self.remove(i);
                return Some(v);
            }
        }
        None
    }
}

impl<T0> VectorUtils<T0> for Vector<T0>
where
    T0: Clone,
{
    fn mapm<T1, C>(self, ctx: &mut C, mut f: impl FnMut(&mut C, T0) -> T1) -> Vector<T1>
    where
        T1: Clone,
    {
        let mut v = Vector::new();
        for x in self {
            let y = f(ctx, x);
            v.push_back(y);
        }
        v
    }

    fn zip<T1>(&self, other: &Vector<T1>) -> Vector<(T0, T1)>
    where
        T1: Clone,
    {
        self.clone()
            .into_iter()
            .zip(other.clone().into_iter())
            .map(|(x, y)| (x, y))
            .collect()
    }

    fn filter_mapm<T1, C>(
        self,
        ctx: &mut C,
        mut f: impl FnMut(&mut C, T0) -> Option<T1>,
    ) -> Vector<T1>
    where
        T1: Clone,
    {
        let mut v = Vector::new();
        for x in self {
            let y = f(ctx, x);
            if let Some(y) = y {
                v.push_back(y);
            }
        }
        v
    }

    fn map<T1>(self, f: impl FnMut(T0) -> T1) -> Vector<T1>
    where
        T1: Clone,
    {
        self.into_iter().map(f).collect()
    }

    fn concat(mut self, other: Vector<T0>) -> Vector<T0> {
        self.append(other);
        self
    }
}

pub trait OrdMapUtils<K0, T0> {
    fn to_vector(self) -> Vector<(K0, T0)>
    where
        K0: Clone,
        T0: Clone;
    fn mapm<T1, F, C>(self, ctx: C, f: F) -> (C, OrdMap<K0, T1>)
    where
        F: Fn(C, T0) -> (C, T1),
        T1: Clone;
}

impl<K0, T0> OrdMapUtils<K0, T0> for OrdMap<K0, T0>
where
    K0: std::hash::Hash + Eq + Clone + std::cmp::Ord,
    T0: Clone,
{
    fn to_vector(self) -> Vector<(K0, T0)>
    where
        K0: Clone,
        T0: Clone,
    {
        self.into_iter().collect()
    }
    fn mapm<T1, F, C>(self, mut ctx: C, f: F) -> (C, OrdMap<K0, T1>)
    where
        F: Fn(C, T0) -> (C, T1),
        T1: Clone,
    {
        let mut map = OrdMap::new();
        for (k, v) in self.into_iter() {
            let (ctx1, v1) = f(ctx, v);
            ctx = ctx1;
            map.insert(k, v1);
        }
        (ctx, map)
    }
}

pub trait OptionUtils<T0> {
    fn mapm<T1, F, C>(self, ctx: &mut C, f: F) -> Option<T1>
    where
        F: FnMut(&mut C, T0) -> T1;

    fn mapm_or_else<T1, F, G, C>(self, ctx: &mut C, f: F, g: G) -> T1
    where
        F: FnMut(&mut C, T0) -> T1,
        G: FnMut(&mut C) -> T1;
}

impl<T0> OptionUtils<T0> for Option<T0> {
    fn mapm<T1, F, C>(self, ctx: &mut C, mut f: F) -> Option<T1>
    where
        F: FnMut(&mut C, T0) -> T1,
    {
        match self {
            None => None,
            Some(x) => {
                let y = f(ctx, x);
                Some(y)
            }
        }
    }

    fn mapm_or_else<T1, F, G, C>(self, ctx: &mut C, mut f: F, mut g: G) -> T1
    where
        F: FnMut(&mut C, T0) -> T1,
        G: FnMut(&mut C) -> T1,
    {
        match self {
            None => g(ctx),
            Some(x) => f(ctx, x),
        }
    }
}
