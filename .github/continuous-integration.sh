#!/bin/bash

set -e # Terminates as soon as something fails

export PATH="$A2M_BUILD/llvm-build/bin:$PATH"

function run-step {
    echo "Running $@"
    "$@"
}

if [[ -d "${PERSIST_DIR}/ccache-cachedir" ]]; then
    echo "The Ccache directory exists at ${PERSIST_DIR}/ccache-cachedir"
else
    echo "Creating Ccache directory at ${PERSIST_DIR}/ccache-cachedir"
    mkdir -p ${PERSIST_DIR}/ccache-cachedir
    envsubst > ${PERSIST_DIR}/ccache-config <<EOF
    max_size = 20G
    cache_dir = ${PERSIST_DIR}/ccache-cachedir
EOF
fi

(
    run-step cd arc-mlir
    A2M_CCACHE="1" BUILD_FLAVOUR=Release A2M_ASSERTS="1" run-step ./arc-mlir-build
    run-step ninja -C $A2M_BUILD/llvm-build/ check-arc-mlir
)

(
    run-step cd arc-script
    run-step cargo insta test --package=arc-script-test-compile
    run-step cargo insta accept
    run-step cargo clippy
    run-step cargo check
    run-step cargo check --bins
    run-step cargo check --tests
    run-step cargo check --examples
    run-step cargo check --benches
    run-step cargo fuzz run parse -- -runs=10000 -only-ascii
    # run-step cargo fmt -- -v --check
)
