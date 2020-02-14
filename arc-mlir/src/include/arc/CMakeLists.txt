include_directories(../../../llvm-project/mlir/include/)
set(LLVM_TARGET_DEFINITIONS Ops.td)
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/")

mlir_tablegen(Ops.h.inc -gen-op-decls "${TBLGEN_INCDIRS}")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "${TBLGEN_INCDIRS}")
mlir_tablegen(Ops.md -gen-op-doc "${TBLGEN_INCDIRS}")

add_public_tablegen_target(ArcDialectOpsIncGen)
