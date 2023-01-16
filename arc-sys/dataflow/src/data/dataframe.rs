use crate::prelude::*;
use macros::Unpin;
use serde::Serialize;
use serde_derive::Deserialize;
use std::rc::Rc;

use std::collections::HashMap;

use super::cell::Cell;
use super::string::Str;

macro_rules! extract {
    ($e:expr, $p:path) => {
        match $e {
            $p(v) => v,
            _ => panic!("unwrap: expected {}", stringify!($p)),
        }
    };
}

#[derive(Serialize, Deserialize, Clone, Debug, Unpin)]
pub struct DataFrame {
    data: Rc<HashMap<String, Column>>,
    head: usize,
    tail: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Unpin)]
enum Column {
    Bool(Vec<Option<bool>>),
    U64(Vec<Option<u64>>),
    I64(Vec<Option<i64>>),
    F64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
}

#[derive(Deserialize, Serialize, Clone, Debug, Unpin)]
#[serde(untagged)]
enum Scalar {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
}

impl DataFrame {
    fn get_mut(&mut self) -> &mut HashMap<String, Column> {
        unsafe { Rc::get_mut_unchecked(&mut self.data) }
    }

    pub fn new() -> Self {
        Self {
            data: Rc::new(HashMap::new()),
            head: 0,
            tail: 0,
        }
    }

    fn read_csv<R: std::io::Read>(mut rdr: csv::Reader<R>) -> Self {
        let headers: Vec<_> = rdr.headers().unwrap().iter().map(String::from).collect();
        let rows: Vec<Vec<Option<Scalar>>> = rdr.deserialize().map(Result::unwrap).collect();
        let columns = transpose(rows).into_iter().map(scalars_to_column);
        let data = headers
            .into_iter()
            .zip(columns)
            .collect::<HashMap<String, Column>>();
        Self {
            data: Rc::new(data),
            head: 0,
            tail: 0,
        }
    }

    pub fn read_csv_file(filepath: Str) -> Self {
        Self::read_csv(csv::Reader::from_path(filepath.as_str()).unwrap())
    }

    pub fn read_csv_data(data: Str) -> Self {
        Self::read_csv(csv::Reader::from_reader(data.as_str().as_bytes()))
    }

    pub fn head(mut self, n: i32) -> Self {
        self.head = n as usize;
        self
    }

    pub fn tail(mut self, n: i32) -> Self {
        self.tail = n as usize;
        self
    }

    fn fold(
        self,
        f_bool: impl Fn(bool, bool) -> bool + Copy,
        f_u64: impl Fn(u64, u64) -> u64 + Copy,
        f_i64: impl Fn(i64, i64) -> i64 + Copy,
        f_f64: impl Fn(f64, f64) -> f64 + Copy,
        f_string: impl Fn(&str, &str) -> String,
    ) -> Self {
        let data = self
            .data
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    Column::U64(v) => Column::U64(fold(&v[self.head..self.tail], 0, f_u64)),
                    Column::I64(v) => Column::I64(fold(&v[self.head..self.tail], 0, f_i64)),
                    Column::F64(v) => Column::F64(fold(&v[self.head..self.tail], 0.0, f_f64)),
                    Column::Bool(v) => Column::Bool(fold(&v[self.head..self.tail], false, f_bool)),
                    Column::String(v) => {
                        Column::String(vec![Some(v.iter().fold(String::new(), |a, x| {
                            f_string(&a, &x.as_ref().map(|x| x.as_str()).unwrap_or(""))
                        }))])
                    }
                };
                (k.clone(), v)
            })
            .collect::<HashMap<_, _>>();
        Self {
            data: Rc::new(data),
            ..self
        }
    }
}

fn fold<T: Copy>(v: &[Option<T>], init: T, f: impl Fn(T, T) -> T) -> Vec<Option<T>> {
    vec![Some(v.iter().map(|x| x.unwrap_or(init)).fold(init, f))]
}

/// Convert a struct into a column entry.
trait AppendRow {
    fn append_row(self, df: DataFrame);
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_i| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

fn scalars_to_column(scalars: Vec<Option<Scalar>>) -> Column {
    match scalars[0].as_ref().unwrap() {
        Scalar::Bool(_) => Column::Bool(
            scalars
                .into_iter()
                .map(|v| v.map(|v| extract!(v, Scalar::Bool)))
                .collect(),
        ),
        Scalar::U64(_) => Column::F64(
            scalars
                .into_iter()
                .map(|v| v.map(|v| extract!(v, Scalar::F64)))
                .collect(),
        ),
        Scalar::I64(_) => Column::I64(
            scalars
                .into_iter()
                .map(|v| v.map(|v| extract!(v, Scalar::I64)))
                .collect(),
        ),
        Scalar::F64(_) => Column::U64(
            scalars
                .into_iter()
                .map(|v| v.map(|v| extract!(v, Scalar::U64)))
                .collect(),
        ),
        Scalar::String(_) => Column::String(
            scalars
                .into_iter()
                .map(|v| v.map(|v| extract!(v, Scalar::String)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct Person {
        name: String,
        age: u64,
        height: f64,
    }

    impl AppendRow for Person {
        fn append_row(self, mut df: DataFrame) {
            extract!(df.get_mut().get_mut("name").unwrap(), Column::String).push(Some(self.name));
            extract!(df.get_mut().get_mut("age").unwrap(), Column::U64).push(Some(self.age));
            extract!(df.get_mut().get_mut("height").unwrap(), Column::F64).push(Some(self.height));
        }
    }

    #[test]
    fn test() {
        // let df = DataFrame::from
    }
}
