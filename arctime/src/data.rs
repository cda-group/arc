use std::fmt::Debug;
pub use time::PrimitiveDateTime as DateTime;

pub trait DataReqs: 'static + Sync + Send + Debug + Clone {}
impl<T> DataReqs for T where T: 'static + Sync + Send + Debug + Clone {}

#[derive(Debug, Clone)]
pub enum Either<L, R> {
    L(L),
    R(R),
}
