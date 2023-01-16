use std::error::Error;
use wasmtime::*;

pub struct Wasm {
    engine: Engine,
    store: Store<()>,
}

impl Wasm {
    fn new() -> Self {
        let engine = Engine::default();
        let store = Store::new(&engine, ());
        Self { engine, store }
    }

    fn run<I: WasmTy, O: WasmTy>(&mut self, source: &[u8], input: I) -> O {
        let module = Module::from_binary(&self.engine, source).unwrap();
        let instance = Instance::new(&mut self.store, &module, &[]).unwrap();
        instance
            .get_func(&mut self.store, "main")
            .unwrap()
            .typed::<I, O, _>(&self.store)
            .unwrap()
            .call(&mut self.store, input)
            .unwrap()
    }
}
