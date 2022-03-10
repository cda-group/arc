#!/bin/bash

set -e # Terminates as soon as something fails

echo "The work dir is ${A2M_BUILD}"

export PATH="$A2M_BUILD/llvm-build/bin:$PATH"
export RUSTC_WRAPPER="/home/arc-runner/.cargo/bin/sccache"
export SCCACHE_DIR="${PERSIST_DIR}/sccache"
export SCCACHE_CACHE_SIZE="20G"
export CARGO_INCREMENTAL="0"

function run-step {
    echo "Running \'$@\'"
    "$@"
}

mkdir -p ${PERSIST_DIR}

if [[ -d "${PERSIST_DIR}/ccache-cachedir" ]]; then
    echo "The Ccache directory exists at ${PERSIST_DIR}/ccache-cachedir"
else
    echo "Creating Ccache directory at ${PERSIST_DIR}/ccache-cachedir"
    mkdir -p ${PERSIST_DIR}/ccache-cachedir
fi

if [[ -f "${CCACHE_CONFIGPATH}" ]]; then
    echo "The Ccache config is:"
    cat "${CCACHE_CONFIGPATH}"
else
    echo "Creating Ccache config at ${CCACHE_CONFIGPATH}"
    envsubst > ${CCACHE_CONFIGPATH} <<EOF
    max_size = 20G
    cache_dir = ${PERSIST_DIR}/ccache-cachedir
EOF
fi

if [[ -d "${SCCACHE_DIR}" ]]; then
    echo "The Sccache directory exists at ${SCCACHE_DIR}"
    echo "It contains $(du -hs ${SCCACHE_DIR} | cut -f1)"
else
    echo "Creating Sccache directory at ${SCCACHE_DIR}"
    mkdir -p ${SCCACHE_DIR}
fi

function check-ccache {
    echo "=== Ccache statistics ==="
    ccache -s
    echo "=== Sccache statistics ==="
    sccache --show-stats
}

function run-build {
    A2M_CCACHE="1" BUILD_FLAVOUR="Release" A2M_ASSERTS="1" run-step ./build
}

function run-mlir-tests {
    run-step ninja -C $A2M_BUILD/llvm-build/ check-arc-mlir
}

function run-runtime-tests {
    cd arc-runtime
    run-step arc-cargo "$@"
}

case $1 in
    check-ccache)
	check-ccache
	;;

    check-cargo)
	check-cargo
	;;

    run-build)
	run-build
	;;

    run-mlir-tests)
	run-mlir-tests
	;;

    cargo)
	# We assume this is a arc-runtime cargo command line
	shift
	run-runtime-tests "$@"
	;;
esac
