add_subdirectory(arc-mlir)

set(CARGO_TOML_TEMPLATE_FILE "${ARC_LANG_SRC_DIR}/etc/Cargo.toml.template")
file(READ ${CARGO_TOML_TEMPLATE_FILE} CARGO_TOML_TEMPLATE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${CARGO_TOML_TEMPLATE_FILE})

configure_file(arc-cargo.in ${LLVM_TOOLS_BINARY_DIR}/arc-cargo @ONLY)
configure_file(arc-mlir-rust-test.in ${LLVM_TOOLS_BINARY_DIR}/arc-mlir-rust-test @ONLY)
configure_file(arc.in ${LLVM_TOOLS_BINARY_DIR}/arc @ONLY)
