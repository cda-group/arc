set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/)
set(MLIR_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/
)

set(LLVM_TARGET_DEFINITIONS ${ARC_MLIR_SOURCE_DIR}/include/Arc/Arc.td)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/")

mlir_tablegen(ArcOps.h.inc -gen-op-decls)
mlir_tablegen(ArcOps.cpp.inc -gen-op-defs)
mlir_tablegen(ArcOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ArcOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArcOpsEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(ArcDialectOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ${ARC_MLIR_SOURCE_DIR}/include/Arc/Opts.td)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/")
mlir_tablegen(ArcOpts.h.inc -gen-rewriters ${TBLGEN_INCDIRS} "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ArcDialectOptsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ArcDialectPassIncGen)


add_mlir_dialect(Arc arc Arc)
