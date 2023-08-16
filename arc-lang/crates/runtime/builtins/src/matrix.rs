#![allow(unused)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use derive_more::Deref;
use derive_more::DerefMut;
use macros::DeepClone;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::IxDynImpl;
use ndarray::OwnedRepr;
use num::Num;
use num::Zero;
use serde::Deserialize;
use serde::Serialize;

use crate::array::Array;
use crate::cow::Cow;
use crate::iterator::Iter;
use crate::traits::Data;
use crate::traits::DeepClone;
use crate::vec::Vec;

#[derive(DeepClone, Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[repr(C)]
pub struct Matrix<T>(pub Cow<Inner<T>>);

#[derive(Debug, Clone, Deref, DerefMut, Serialize, Deserialize, Eq, PartialEq)]
#[repr(C)]
pub struct Inner<T>(pub ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>);

impl<T: Clone> DeepClone for Inner<T> {
    fn deep_clone(&self) -> Self {
        Inner(self.0.clone())
    }
}

impl<T> Matrix<T> {
    pub fn new<const N: usize>(shape: impl Into<Array<usize, N>>) -> Self
    where
        T: Clone + Zero,
    {
        Matrix::from(ArrayBase::zeros(shape.into().0.to_vec()))
    }

    pub fn insert_axis(mut self, axis: usize) -> Self
    where
        T: Clone,
    {
        self.0.update(|this| this.0.insert_axis_inplace(Axis(axis)));
        self
    }

    pub fn remove_axis(mut self, axis: usize) -> Self
    where
        T: Clone,
    {
        self.0
            .map(|this| Matrix::from(this.0.remove_axis(Axis(axis))))
    }

    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.0.map(|this| Vec::from(this.0.into_raw_vec()))
    }

    pub fn iter(self) -> Iter<T, impl Iterator<Item = T> + Clone>
    where
        T: Clone,
    {
        Iter::new(self.0.take().0.into_raw_vec().into_iter())
    }
}

impl<T> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for Matrix<T> {
    fn from(array: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        Matrix(Cow::new(Inner(array)))
    }
}
