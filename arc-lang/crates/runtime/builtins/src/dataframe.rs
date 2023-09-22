// use crate::cow::Cow;
// use crate::vec::Vec;
//
// pub struct DataFrame(pub Cow<polars::prelude::DataFrame>);
//
// pub struct Series(pub Cow<polars::prelude::Series>);
//
// impl DataFrame {
//     pub fn new(columns: Vec<Series>) -> Self {
//         DataFrame(polars::prelude::DataFrame::new(columns.0.take()))
//     }
//
// }
