#!/bin/sh

check () {
  cd "$1";
  echo "[$1]";
  echo "[cargo check]"; cargo check;
  echo "[cargo test]"; cargo test;
  echo "[cargo clippy]"; cargo clippy;
  echo "[cargo fmt]"; cargo fmt -- --check;
  echo "[cargo rustc]"; cargo rustc --lib -- -D warnings;
  cd -
}

for crate in arc-script-*; do
  check "$crate"
done

check "."
