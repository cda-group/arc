file(GLOB globbed *.c *.cpp)
add_mlir_dialect_library(ArcDialect
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ArcOps

  DEPENDS
  MLIRArcDialectIncGen
  ArcDialectOptsIncGen
)

set(LLVM_TARGET_DEFINITIONS ${ARC_MLIR_SOURCE_DIR}/include/Arc/Opts.td)
set(TBLGEN_INCDIRS
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/"
  "-I${CMAKE_CURRENT_SOURCE_DIR}/../"
)
mlir_tablegen(Opts.inc -gen-rewriters ${TBLGEN_INCDIRS})
add_public_tablegen_target(ArcDialectOptsIncGen)

target_link_libraries(ArcDialect
  PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  LLVMSupport
  )
