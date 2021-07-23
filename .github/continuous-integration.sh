#!/bin/bash

set -e # Terminates as soon as something fails

echo "The work dir is ${A2M_BUILD}"

export PATH="$A2M_BUILD/llvm-build/bin:$PATH"

function run-step {
    echo "Running \'$@\'"
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

function check-ccache {
    echo "Ccache statistics:"
    ccache -s
}

function run-mlir-build {
    cd arc-mlir
    A2M_CCACHE="1" BUILD_FLAVOUR="Release" A2M_ASSERTS="1" \
	      run-step ./arc-mlir-build
}

function run-mlir-tests {
    run-step ninja -C $A2M_BUILD/llvm-build/ check-arc-mlir
}

function run-arc-script-test {
    cd arc-script
    run-step arc-cargo "$@"
}

case $1 in
    check-ccache)
	check-ccache
	;;

    check-cargo)
	check-cargo
	;;

    run-mlir-build)
	run-mlir-build
	;;

    run-mlir-tests)
	run-mlir-tests
	;;

    cargo)
	# We assume this is a arc-script cargo command line
	shift
	run-arc-script-test "$@"
	;;
esac
