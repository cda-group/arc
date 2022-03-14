#!/bin/bash

set -e

usage () {
  echo "Usage: $0 [OPTIONS] [check|build|run] <INPUT-FILE> [-- <MLIR-OPTS>]"
  echo "Options:"
  echo "  -h          Show this help message"
  echo "  -o <DIR>    Output crate directory"
  echo "  -t <DIR>    Output crate target directory"
  echo "  -r          Compile in release mode"
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

CARGO_FLAGS=()

while getopts ho:t:r opt; do
    case $opt in
        h)
            usage
            ;;
        o)
            CRATE_DIR="$OPTARG"
            ;;
        t)
            CRATE_TARGET_DIR="$OPTARG"
            ;;
        r)
            COMPILE_MODE="release"
            CARGO_FLAGS+=("--release")
            ;;
        *)
            echo "Invalid option: $OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "$COMPILE_MODE" ]; then
    COMPILE_MODE="debug"
fi

shift "$((OPTIND - 1))"

if [ $# -gt 0 ]; then
    case $1 in
        check|build|run)
            CARGO_MODE="$1"
            ;;
        *)
            echo "Invalid subcommand: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
else
    echo "No subcommand specified" >&2
    usage
fi

debug "CARGO_MODE: $CARGO_MODE"

READLINK="readlink"
if command -v greadlink > /dev/null; then
  READLINK="greadlink" # Prefer greadlink over readlink
fi

debug "READLINK: $READLINK"

if [[ $# -gt 0 ]]; then
    export INPUT_FILE="$($READLINK -f "$1")"
    shift
else
    echo "No input file specified" >&2
    usage
    exit 1
fi

debug "INPUT_FILE: $INPUT_FILE"

if [[ $# -gt 0 ]]; then
    case $1 in
        --)
            shift
            ;;
        *)
            echo "Found unexpected argument: $1" >&2
            usage
            exit 1
            ;;
    esac
fi

debug "MLIR args: $@"

# Find root directory (borrowed from https://github.com/frej/fast-export/blob/dbb8158527719e67874e0bba37970e48e1aaedfb/hg-fast-export.sh#L6)
if ! $READLINK -f "$(which "$0")" > /dev/null 2>&1; then
    ROOT="$(dirname "$(which "$0")")"
else
    ROOT="$(dirname "$($READLINK -f "$(which "$0")")")"
fi

debug "ROOT: $ROOT"

export CRATE_NAME="$(basename "$INPUT_FILE" .arc)-crate"
debug "CRATE_NAME: $CRATE_NAME"

if [ -z "$INPUT_FILE" ]; then
    usage
fi

debug "INPUT_FILE: $INPUT_FILE"

if [ -z "$CRATE_DIR" ]; then
    CRATE_DIR="$(dirname "$INPUT_FILE")/$CRATE_NAME"
fi

if [ -z "$CRATE_TARGET_DIR" ]; then
    CRATE_TARGET_DIR="$CRATE_DIR/target"
fi

CRATE_MAIN_FILE="$CRATE_DIR/src/main.rs"
CRATE_TOML_FILE="$CRATE_DIR/Cargo.toml"
CRATE_MLIR_FILE="$CRATE_DIR/src/main.mlir"

debug "CRATE_DIR: $CRATE_DIR"
debug "CRATE_MAIN_FILE: $CRATE_MAIN_FILE"
debug "CRATE_TOML_FILE: $CRATE_TOML_FILE"

CARGO_FLAGS+=("--manifest-path=$CRATE_TOML_FILE")
CARGO_FLAGS+=("--target-dir=$CRATE_TARGET_DIR")

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
    export ARC_RUNTIME_DIR="$ROOT/arc-runtime/"
fi

debug "ARC_RUNTIME_DIR: $ARC_RUNTIME_DIR"

mkdir -p "$CRATE_DIR/src"

# Create the Cargo.toml
envsubst > "$CRATE_TOML_FILE" <<'EOF'
[package]
name    = "$CRATE_NAME"
version = "0.0.0"
edition = "2021"

[dependencies]
arc-runtime = { version = "=0.0.0", path = "$ARC_RUNTIME_DIR" }
hexf        = { version = "0.2.1" }
serde       = { version = "1.0.136" }

[profile.dev]
opt-level = 0
split-debuginfo = "unpacked"
EOF

[ "$ARC_DEBUG" ] && echo "$CRATE_TOML_FILE: " && cat "$CRATE_TOML_FILE"

envsubst > "$CRATE_MAIN_FILE" <<'EOF'
// Generated source for $INPUT_FILE
#![feature(unboxed_closures)]
#![feature(imported_main)]
EOF

"$ARC_LANG" "$INPUT_FILE" > "$CRATE_MLIR_FILE"
  
[ "$ARC_DEBUG" ] && echo "$CRATE_MLIR_FILE: " && cat "$CRATE_MLIR_FILE"

"$ARC_MLIR" "$CRATE_MLIR_FILE" "$@" -arc-to-rust -inline-rust -arc-lang-runtime >> "$CRATE_MAIN_FILE"

echo 'fn main() { toplevel::main() }' >> "$CRATE_MAIN_FILE"

[ "$ARC_DEBUG" ] && echo "$CRATE_MAIL_FILE: " && cat "$CRATE_MAIN_FILE"

debug "CARGO_FLAGS: $CARGO_FLAGS"

cargo "$CARGO_MODE" "${CARGO_FLAGS[@]}"
