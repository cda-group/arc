#!/bin/bash

if [ $# -eq 1 ]
then
  git rebase "$1" \
    --exec="cd arc-script && cargo check --all-features --bins --examples --tests --benches" \
    --exec="cd arc-script && cargo test" \
    --exec="cd arc-script && cargo clippy" \
    --exec="cd arc-script && cargo fmt -- --check" \
    --exec="cd arc-script/core && cargo rustc --lib -- -D warnings" \
    --exec="cd arc-script/wasm && cargo rustc --lib -- -D warnings" \
    --exec="cd arc-script/macros && cargo rustc --lib -- -D warnings"
else
  echo "Expected branch as argument."
fi
