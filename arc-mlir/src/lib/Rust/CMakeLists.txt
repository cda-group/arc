file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(RustDialect
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/RustOps

  DEPENDS
  MLIRRustIncGen
)

target_link_libraries(RustDialect
  PUBLIC
  MLIRIR
  MLIRFuncDialect
  LLVMSupport
)
