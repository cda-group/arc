#!/bin/bash

set -e # Terminates as soon as something fails

echo "The work dir is ${ARC_MLIR_BUILD}"

export PATH="$ARC_MLIR_BUILD/llvm-build/bin:$PATH"
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
    ARC_MLIR_CCACHE="1" BUILD_FLAVOUR="Release" ARC_MLIR_ASSERTS="1" run-step ./build
}

function run-mlir-tests {
    run-step ninja -C $ARC_MLIR_BUILD/llvm-build/ check-arc-mlir
}

function run-sys-tests {
    cd arc-sys
    run-step arc-cargo "$@"
}

function install-cmake {
    wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
    tar -xf cmake-3.23.1.tar.gz
    cd cmake-3.23.1
    ./configure
    make
    make install
}

function install-rust {
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
}

function install-ocaml {
    opam init -y --disable-sandboxing
    eval $(opam config env)
    opam install -y dune core menhir
}

function install-ubuntu-packages {
    sudo apt update
    sudo apt install software-properties-common
    add-apt-repository ppa:avsm/ppa
    sudo apt update
    sudo apt install -y clang libssl-dev ninja-build wget make curl opam
    (install-ocaml)
    (install-cmake)
}

function install-macos-packages {
    (install-ocaml)
    (install-cmake)
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

    install-ubuntu-packages)
  install-ubuntu-packages
  ;;

    install-macos-packages)
  install-macos-packages
  ;;

    cargo)
	# We assume this is a arc-sys cargo command line
	shift
	run-sys-tests "$@"
	;;
esac
