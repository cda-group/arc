#!/bin/bash

if [ $# -eq 1 ]
then
  git rebase "$1" \
    --exec="cd arc-script; cargo check --all-features --bins --examples --tests --benches; cd -" \
    --exec="cd arc-script; cargo test; cd -" \
    --exec="cd arc-script; cargo clippy; cd -" \
    --exec="cd arc-script; cargo fmt -- --check; cd -" \
    --exec="cd arc-script/core && cargo rustc --lib -- -D warnings; cd -" \
    --exec="cd arc-script/wasm && cargo rustc --lib -- -D warnings; cd -" \
    --exec="cd arc-script/macros && cargo rustc --lib -- -D warnings; cd -"
else
  echo "Expected branch as argument."
fi
