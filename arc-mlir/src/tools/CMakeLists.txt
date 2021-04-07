add_subdirectory(arc-mlir)

configure_file(arc-cargo.in ${LLVM_TOOLS_BINARY_DIR}/arc-cargo @ONLY)
configure_file(arc-mlir-rust-test.in ${LLVM_TOOLS_BINARY_DIR}/arc-mlir-rust-test @ONLY)
