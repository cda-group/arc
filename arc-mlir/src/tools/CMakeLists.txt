add_subdirectory(arc-mlir)

# Generate script files from a templates (e.g., arc-cargo from arc-cargo.in) by
# replacing variables like @VAR@ with their CMake values. The resulting file 
# will be placed in the LLVM_TOOLS_BINARY_DIR directory. @ONLY indicates that 
# only variables of the form @VAR@ will be replaced, not cmake built-in 
# substitutions.
configure_file(arc-cargo.in ${LLVM_TOOLS_BINARY_DIR}/arc-cargo @ONLY)
configure_file(arc-mlir-rust-test.in ${LLVM_TOOLS_BINARY_DIR}/arc-mlir-rust-test @ONLY)
configure_file(arc.in ${LLVM_TOOLS_BINARY_DIR}/arc @ONLY)
