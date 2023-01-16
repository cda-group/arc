use std::io::Write;
use std::process::Command;
use std::process::Stdio;

use halfbrown::HashMap;
use regex::Regex;
use shared::api::Graph;
use shared::api::Node;

pub struct Graph1 {
    pub code: String,
    pub nodes: HashMap<String, Node>,
}

/// Compile MLIR source code into Rust and extract the dataflow graph.
pub fn lower(source: &str) -> Graph1 {
    let mut cmd = Command::new(env!("ARC_MLIR_CMD"))
        .arg("-arc-to-rust")
        .arg("-inline-rust")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to execute process");

    let stdin = cmd.stdin.as_mut().unwrap();
    stdin.write_all(source.as_bytes()).unwrap();
    drop(stdin);

    let output = cmd
        .wait_with_output()
        .expect("Failed to read stdout")
        .stdout;

    let code = std::str::from_utf8(&output).unwrap().to_owned();
    let json = extract_json(&code);
    let graph = serde_json::from_str::<Graph>(json).unwrap();
    Graph1 {
        code,
        nodes: graph.nodes.into_iter().collect(),
    }
}

fn extract_json(rust: &str) -> &str {
    let regex = Regex::new(
        r"(?xms) # x: Enable insignificant whitespace
                 # m: Enable multi-line mode 
                 # s: Enable matching \n with .

          ^JSON_START_MARKER$

          (?P<json>.*)       # JSON code

          ^JSON_END_MARKER$
        ",
    )
    .unwrap();
    let captures = regex.captures(rust).unwrap();
    captures.name("json").unwrap().as_str()
}

#[test]
fn test_regex() {
    let source = indoc::indoc! {
    r#"
        fn map(x: i32) -> i32 {
            x + 1
        }
        /*
        JSON_START_MARKER
        { "hello": "world" }
        JSON_END_MARKER
        */
    "#};

    let json = extract_json(source);

    assert_eq!(
        json.split_whitespace().collect::<String>(),
        r#"{ "hello": "world" }"#.split_whitespace().collect::<String>()
    );
}

#[test]
fn test_serde() {
    let graph = Graph {
        nodes: vec![
            (
                "x0".to_string(),
                Node::Source {
                    key_type: "i32".to_string(),
                    data_type: "i32".to_string(),
                    topic: "topic".to_string(),
                    num_partitions: 1,
                },
            ),
            (
                "x1".to_string(),
                Node::Map {
                    input: "x0".to_string(),
                    fun: "map".to_string(),
                },
            ),
            (
                "x2".to_string(),
                Node::Sink {
                    input: "map".to_string(),
                    topic: "topic".to_string(),
                    fun: "x1".to_string(),
                },
            ),
        ]
        .into_iter()
        .collect(),
    };

    let json = serde_json::to_string(&graph).unwrap();

    assert_eq!(
        json.split_whitespace().collect::<String>(),
        r#"
            {
                "nodes": {
                    "x0": { "Source": { "key_type": "i32", "data_type": "i32", "topic": "topic", "num_partitions": 1 } },
                    "x1": { "Map": { "input": "x0", "fun": "map" } },
                    "x2": { "Sink": { "input": "map", "topic": "topic", "fun": "x1" } }
                }
            }
        "#
        .split_whitespace()
        .collect::<String>()
    )
}
