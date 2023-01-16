use itertools::multiunzip;
use itertools::Itertools;
use proc_macro2::Ident;
use proc_macro2::Literal;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use quote::quote;
use rust_format::Formatter;

use crate::server::ServerConfig;
use crate::server::WorkerId;

use super::lowering2::Node2;
use super::lowering2::NodeId;
use super::lowering5::Graph5;
use super::lowering5::Shard4;

#[derive(Debug)]
pub struct Graph6 {
    pub code: String,
    pub shards: Vec<Shard4>,
}

pub(crate) fn lower(graph: Graph5, config: &ServerConfig) -> Graph6 {
    let (worker_defs, worker_ids): (Vec<_>, Vec<_>) = graph
        .shards
        .iter()
        .group_by(|shard| shard.worker_id)
        .into_iter()
        .map(|(worker_id, shards)| compile_worker(worker_id, shards, &graph, config))
        .unzip();
    let worker_id_strings = worker_ids.iter().map(|i| i.to_string()).collect::<Vec<_>>();
    let code: TokenStream = graph.code.parse().unwrap();
    let main = quote! {
        use dataflow::prelude::*;

        fn main() {
            let mut args = std::env::args();
            let _ = args.next();
            let kind = args.next();
            let db = Database::remote("127.0.0.1:2379");
            match kind.unwrap().as_str() {
                #(
                    #worker_id_strings => #worker_ids(db)
                ,)*
                _ => panic!("Unknown arg, expected instance[N]"),
            }
        }

        #(#worker_defs)*

        #code
    };
    Graph6 {
        code: rust_format::RustFmt::default().format_tokens(main).unwrap(),
        shards: graph.shards,
    }
}

pub fn compile_worker<'a>(
    worker_id: WorkerId,
    shards: impl Iterator<Item = &'a Shard4>,
    graph: &Graph5,
    config: &ServerConfig,
) -> (TokenStream, Ident) {
    let worker_id = new_worker_id(worker_id);
    let (shard_defs, shard_ids, shard_cpus): (Vec<_>, Vec<_>, Vec<_>) =
        multiunzip(shards.map(|s| compile_shard(s, graph, config)));
    let instance_def = quote! {
        fn #worker_id(db: Database) {
            Runtime::new()#(.spawn(#shard_ids(db.clone()), #shard_cpus))*;
        }
        #(#shard_defs)*
    };
    (instance_def, worker_id)
}

pub fn compile_shard(
    shard: &Shard4,
    graph: &Graph5,
    config: &ServerConfig,
) -> (TokenStream, Ident, usize) {
    let pipeline = graph.pipelines.get(&shard.pipeline_id).unwrap();
    let mut pre_loop_stmts = Vec::new();
    let mut loop_head_stmts = Vec::new();
    let mut loop_tail_stmts = Vec::new();
    let mut param_ids = Vec::new();
    let mut state_ids = Vec::new();
    let mut source_id = None;
    let mut operator_index = 0;
    let shard_id = new_shard_id(shard);
    let shard_id_string = shard_id.to_string();
    for (id, node) in &pipeline.nodes {
        let node_id = new_node_id(id);
        match node {
            _ if node.is_source() => {
                source_id = Some(node_id.clone());
                let init_expr = new(id, node, shard, graph, config);
                pre_loop_stmts.push(quote!(let mut #node_id = #init_expr;));
            }
            _ if node.is_sink() => {
                let init_expr = new(id, node, shard, graph, config);
                let send_expr = send(node, node_id.clone());
                loop_head_stmts.push(quote!(#send_expr;));
                pre_loop_stmts.push(quote!(let mut #node_id = #init_expr;));
            }
            _ => {
                let node_index = new_index(operator_index);
                let param_id = new_param_id(operator_index);
                let state_id = new_state_id(operator_index);
                let init_expr = new(id, node, shard, graph, config);
                let process_expr = process(node, node_index.clone());
                param_ids.push(param_id.clone());
                state_ids.push(state_id.clone());
                pre_loop_stmts.push(quote!(let (#param_id, #state_id) = #init_expr;));
                loop_head_stmts.push(quote!(let mut #node_id = #process_expr;));
                loop_tail_stmts.push(quote!(state.#node_index = #node_id.state();));
                operator_index += 1;
            }
        }
    }
    loop_tail_stmts.reverse();
    let source_id = source_id.expect("no source node");
    let shard_def = quote! {
        async fn #shard_id(db: Database) {
            #(#pre_loop_stmts)*
            let param = (#(#param_ids,)*);
            let mut state = State::new(#shard_id_string, db, (#(#state_ids,)*));
            while let Some((key, mut #source_id)) = #source_id.recv().await {
                let mut state = state.get(key);
                #(#loop_head_stmts)*
                #(#loop_tail_stmts)*
            }
        }
    };
    (shard_def, shard_id, shard.cpu)
}

fn new(
    id: &NodeId,
    node: &Node2,
    shard: &Shard4,
    graph: &Graph5,
    config: &ServerConfig,
) -> TokenStream {
    match node {
        Node2::KafkaSource {
            key_type,
            data_type,
            topic,
            num_partitions,
        } => {
            let broker = config.broker.to_string();
            let key_type = parse_code(key_type);
            let data_type = parse_code(data_type);
            let num_partitions = *num_partitions as i32;
            quote!(KafkaSource::<#key_type, #data_type>::new(#broker, #topic, 0..#num_partitions))
        }
        Node2::KafkaSink { topic, fun, .. } => {
            let broker = config.broker.to_string();
            let fun = parse_code(fun);
            quote!(KafkaSink::new(#broker, #topic, #fun))
        }
        Node2::ShuffleSource { fun, .. } => {
            let parallellism = graph.parallelism;
            let fun = parse_code(fun);
            let port = shard.source_ports.get(&id);
            quote!(ShuffleSource::new(#port, #parallellism, #fun).await)
        }
        Node2::ShuffleSink { fun, .. } => {
            let fun = parse_code(fun);
            let addrs = graph
                .sink_ports
                .get(id)
                .unwrap()
                .iter()
                .map(|addr| addr.to_string());
            quote!(ShuffleSink::new([#(#addrs),*], #fun).await)
        }
        Node2::Filter { fun, .. } => {
            let fun = parse_code(fun);
            quote!(Filter::new(#fun))
        }
        Node2::Map { fun, .. } => {
            let fun = parse_code(fun);
            quote!(Map::new(#fun))
        }
        Node2::Union { .. } => {
            quote!(Union::new())
        }
        Node2::Window { .. } => {
            quote!(Window::new())
        }
    }
}

fn process(node: &Node2, node_index: Literal) -> TokenStream {
    match node {
        Node2::Filter { input, .. } => {
            let input = new_node_id(input);
            quote!(Filter::process(&mut #input, param.#node_index, state.#node_index))
        }
        Node2::Map { input, .. } => {
            let input = new_node_id(input);
            quote!(Map::process(&mut #input, param.#node_index, state.#node_index))
        }
        Node2::Union { input0, input1, .. } => {
            let input0 = new_node_id(input0);
            let input1 = new_node_id(input1);
            quote!(Union::process(&mut #input0, &mut #input1, param.#node_index, state.#node_index))
        }
        Node2::Window { input } => {
            let input = new_node_id(input);
            quote!(Window::process(&mut #input))
        }
        Node2::KafkaSource { .. }
        | Node2::KafkaSink { .. }
        | Node2::ShuffleSource { .. }
        | Node2::ShuffleSink { .. } => unreachable!(),
    }
}

fn send(node: &Node2, node_id: Ident) -> TokenStream {
    match node {
        Node2::KafkaSink { input, .. } | Node2::ShuffleSink { input, .. } => {
            let input = new_node_id(input);
            quote!(#node_id.send(&mut #input).await)
        }
        Node2::ShuffleSource { .. }
        | Node2::Filter { .. }
        | Node2::Map { .. }
        | Node2::KafkaSource { .. }
        | Node2::Union { .. }
        | Node2::Window { .. } => unreachable!(),
    }
}

fn new_id(s: &str) -> Ident {
    Ident::new(s, Span::call_site())
}

fn new_node_id(id: &NodeId) -> Ident {
    new_id(&format!("node{}", id.0))
}

fn new_param_id(id: usize) -> Ident {
    new_id(&format!("param{}", id))
}

fn new_state_id(id: usize) -> Ident {
    new_id(&format!("state{}", id))
}

fn new_index(i: usize) -> Literal {
    Literal::usize_unsuffixed(i)
}

fn parse_code(c: &String) -> TokenStream {
    c.parse().unwrap()
}

fn new_worker_id(id: WorkerId) -> Ident {
    new_id(&format!("worker{}", id.0))
}

fn new_shard_id(shard: &Shard4) -> Ident {
    new_id(&format!("worker{}_shard{}", shard.worker_id.0, shard.cpu))
}
