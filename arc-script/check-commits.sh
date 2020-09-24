#!/bin/bash

git rebase mlir \
  --exec="cd arc-script && cargo check --all-features --bins --examples --tests --benches" \
  --exec="cd arc-script && cargo test" \
  --exec="cd arc-script && cargo clippy" \
  --exec="cd arc-script && cargo fmt -- --check"

