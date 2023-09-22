use ndarray::ArrayBase;
use ndarray::CowArray;
use ndarray::CowRepr;
use ndarray::Dim;
use ndarray::IxDynImpl;
use once_cell::sync::Lazy;
use ort::tensor::IntoTensorElementDataType;
use ort::tensor::TensorDataToType;
use ort::value::DynArrayRef;
use ort::Environment;
use ort::InMemorySession;
use ort::SessionBuilder;
use ort::Value;
use serde::Serialize;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use crate::blob::Blob;
use crate::matrix::Matrix;

#[derive(Clone)]
#[repr(C)]
pub struct Model {
    bytes: &'static [u8],
    session: Rc<InMemorySession<'static>>,
}

impl Serialize for Model {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(self.bytes)
    }
}

impl<'de> serde::Deserialize<'de> for Model {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes = <std::vec::Vec<u8>>::deserialize(d)?;
        let bytes = std::vec::Vec::leak(bytes);
        Ok(Model::from_bytes(bytes))
    }
}

impl Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

fn init() -> Arc<Environment> {
    Environment::builder()
        .build()
        .expect("Failed building ONNX")
        .into_arc()
}

static ONNX: Lazy<Arc<Environment>> = Lazy::new(init);

impl Model {
    pub fn new(blob: Blob) -> Self {
        Self::from_bytes(blob.0.take().leak())
    }

    fn from_bytes(bytes: &'static [u8]) -> Self {
        let session = SessionBuilder::new(&ONNX)
            .unwrap()
            .with_model_from_memory(bytes)
            .unwrap();
        Model {
            bytes,
            session: Rc::new(session),
        }
    }

    pub fn predict<I, O>(&self, x: Matrix<I>) -> Matrix<O>
    where
        for<'a> DynArrayRef<'a>: From<ArrayBase<CowRepr<'a, I>, Dim<IxDynImpl>>>,
        I: IntoTensorElementDataType + Debug + Clone,
        O: TensorDataToType + Clone,
    {
        let x = x.0.map(|x| CowArray::from(x.0));
        let x = Value::from_array(self.session.allocator(), &x).unwrap();
        let y = self.session.run(vec![x]).unwrap();
        let y = y[0].try_extract::<O>().unwrap();
        let y = y.view().map(|x| x.clone());
        Matrix::from(y)
    }
}
