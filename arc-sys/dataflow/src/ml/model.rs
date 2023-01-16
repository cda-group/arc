use std::path::PathBuf;
use std::{error::Error, marker::PhantomData};

use tensorflow::{
    eager, ops::ReadFile, Graph, ImportGraphDefOptions, SavedModelBundle, Session, SessionOptions,
    SessionRunArgs, Tensor, TensorType, DEFAULT_SERVING_SIGNATURE_DEF_KEY,
};

struct Model<X, Y> {
    bundle: SavedModelBundle,
    graph: Graph,
    marker: PhantomData<(X, Y)>,
}

impl<X: TensorType, Y: TensorType> Model<X, Y> {
    fn new(dir: &str) -> Self {
        let dir = PathBuf::from(dir);
        let opts = eager::ContextOptions::new();
        let ctx = eager::Context::new(opts).unwrap();
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, dir)
            .expect("Failed to load model");

        Model {
            bundle,
            graph,
            marker: PhantomData,
        }
    }

    fn infer(&self, x: Tensor<X>) -> Tensor<Y> {
        let signature = self
            .bundle
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .expect("Failed to get signature");
        let x_info = signature.inputs().values().next().unwrap();
        let y_info = signature.outputs().values().next().unwrap();
        let op_x = self
            .graph
            .operation_by_name_required(&x_info.name().name)
            .expect("Failed to get op");
        let op_y = self
            .graph
            .operation_by_name_required(&y_info.name().name)
            .expect("Failed to get op");

        let mut args = SessionRunArgs::new();
        args.add_feed(&op_x, 0, &x);
        let token_y = args.request_fetch(&op_y, 0);
        self.bundle
            .session
            .run(&mut args)
            .expect("Failed to run session");
        let y: Tensor<Y> = args.fetch(token_y).expect("Failed to fetch output");
        y
    }
}
