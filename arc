#!/bin/bash

set -e

usage () {
  echo "Usage: $0 [OPTIONS] [check|build|run] <INPUT-FILE> [-- <MLIR-OPTS>]"
  echo "Options:"
  echo "  -h       Show this help message"
  echo "  -o <DIR> Output directory (defaults to ./target/)"
  exit 0
}

if [ "$ARC_DEBUG" ]; then
  debug () {
    echo "$@"
  }
else
  debug () {
    :
  }
fi

while getopts ho: opt; do
    case $opt in
        h)
            usage
            ;;
        o)
            CRATE_DIR="$OPTARG"
            ;;
        \?)
            echo "Invalid option: $OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

shift "$((OPTIND - 1))"

debug "Parsed options"

while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        check)
            CARGO_MODE=check
            INPUT_FILE="$(realpath $2)"
            shift 3
            break
            ;;
        build)
            CARGO_MODE=build
            INPUT_FILE="$(realpath $2)"
            shift 3
            break
            ;;
        run)
            CARGO_MODE=run
            INPUT_FILE="$(realpath $2)"
            shift 3
            break
            ;;
        *)
            echo "Invalid subcommand: $1" >&2
            usage
            exit 1
            ;;
    esac
done

debug "Parsed subcommand"
debug "MLIR args: $@"

# Find root directory (borrowed from https://github.com/frej/fast-export/blob/dbb8158527719e67874e0bba37970e48e1aaedfb/hg-fast-export.sh#L6)
READLINK="readlink"
if command -v greadlink > /dev/null; then
  READLINK="greadlink" # Prefer greadlink over readlink
fi

debug "READLINK: $READLINK"

if ! $READLINK -f "$(which "$0")" > /dev/null 2>&1; then
    ROOT="$(dirname "$(which "$0")")"
else
    ROOT="$(dirname "$($READLINK -f "$(which "$0")")")"
fi

debug "ROOT: $ROOT"

CRATE_NAME="$(basename "$INPUT_FILE" .arc)-crate"
debug "CRATE_NAME: $CRATE_NAME"

if [ -z "$INPUT_FILE" ]; then
    usage
fi

debug "INPUT_FILE: $INPUT_FILE"

if [ -z "$CRATE_DIR" ]; then
    CRATE_DIR="$(dirname "$INPUT_FILE")/$CRATE_NAME"
fi

CRATE_MAIN_FILE="$CRATE_DIR/src/main.rs"
CRATE_TOML_FILE="$CRATE_DIR/Cargo.toml"

debug "CRATE_DIR: $CRATE_DIR"
debug "CRATE_MAIN_FILE: $CRATE_MAIN_FILE"
debug "CRATE_TOML_FILE: $CRATE_TOML_FILE"

if [ -z "$ARC_LANG" ]; then
    ARC_LANG="$ROOT/arc-mlir/build/llvm-build/bin/arc-lang"
    if [ ! -f "$ARC_LANG" ]; then
        echo "Could not find arc-lang. Did you run \'$ARC_DIR./build\'?"
        exit 1
    fi
fi

debug "ARC_LANG: $ARC_LANG"

if [ -z "$ARC_MLIR" ]; then
    ARC_MLIR="$ROOT/arc-mlir/build/llvm-build/bin/arc-mlir"
    if [ ! -f "$ARC_LANG" ]; then
        echo "Could not find arc-mlir. Did you run './build'?"
        exit 1
    fi
fi

debug "ARC_MLIR: $ARC_MLIR"

if [ -z "$ARC_RUNTIME_DIR" ]; then
    ARC_RUNTIME_DIR="$ROOT/arc-runtime/"
fi

debug "ARC_RUNTIME_DIR: $ARC_RUNTIME_DIR"

mkdir -p "$CRATE_DIR/src"

# Create the Cargo.toml
envsubst > "$CRATE_DIR/Cargo.toml" <<'EOF'
[package]
name        = "$CRATE_NAME"
version     = "0.0.0"
edition     = "2021"
license     = "MIT"
description = "Generic MLIR-to-Rust test case"

[dependencies]
arc-runtime = { version = "=0.0.0", path = "$ARC_RUNTIME_DIR", features = ["legacy"] }
hexf        = { version = "0.2.1" }
ndarray     = { version = "0.13.0" }
prost       = { version = "0.7.0" }

[profile.dev]
debug = false
EOF

debug "Cargo.toml: $(cat "$CRATE_DIR/Cargo.toml")"

echo '#![feature(unboxed_closures)]' > "$CRATE_MAIN_FILE"
"$ARC_LANG" "$INPUT_FILE" | "$ARC_MLIR" "$@" -arc-to-rust -inline-rust -arc-lang-runtime >> "$CRATE_MAIN_FILE"

case "$CARGO_MODE" in
    check)
        cargo check --manifest-path "$CRATE_TOML_FILE"
        ;;
    build)
        cargo build --manifest-path "$CRATE_TOML_FILE"
        ;;
    run)
        cargo run --manifest-path "$CRATE_TOML_FILE"
        ;;
esac

[ "$ARC_DEBUG" ] && echo "Generated code: " && cat "$CRATE_MAIN_FILE"
