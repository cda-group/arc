#!/bin/sh

check () {
  cd "$1";
  echo "[$1]";
  echo "[cargo check]"; cargo check --quiet;
  echo "[cargo test]"; cargo test --quiet;
  echo "[cargo clippy]"; cargo clippy --quiet;
  echo "[cargo fmt]";   cargo fmt --quiet -- --check;
  echo "[cargo rustc]"; cargo rustc --quiet --lib -- -D warnings;
  cd -
}

for crate in arc-script-*; do
  check "$crate"
done

check "."
