file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(ArcDialect
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ArcOps

  DEPENDS
  MLIRArcIncGen
  ArcDialectOptsIncGen
  ArcDialectOpsEnumsIncGen
  ArcDialectPassIncGen
)

target_link_libraries(ArcDialect
  PUBLIC
  MLIRIR
  MLIRFuncDialect
  LLVMSupport
  )
