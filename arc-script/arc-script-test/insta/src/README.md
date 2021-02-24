# Snapshot-tests

Snapshot-tests record the relation between inputs and outputs to and from the 
compiler. Inputs are in this case arc-script sources and outputs are either 
error diagnostics or their translation into AST, HIR, DFG, MLIR, and Rust 
sources. Each folder represents a different kind of test:

* `./expect-ice/` are tests which are expected to make the compiler crash due to bugs.
* `./expect-pass/` are tests which are expected to pass and emit correct sources.
* `./expect-fail/` are tests which are expected to fail and emit correct diagnostics.
* `./expect-fail-todo` are tests which are expected to pass but currently fail with errors.
* `./expect-mlir-fail-todo` are tests which do not yet compile to MLIR but are expected to work for all other outputs.
* `./snapshots` store the latest version of all snapshots. CI will fail if there is a mismatch between these and the output of insta.

To run snapshot tests and review their results, install `cargo-insta` with 
`cargo install cargo-insta`, and then do `cargo insta test; cargo insta review`.
