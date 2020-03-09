include_directories(../../../llvm-project/mlir/include/)
set(LLVM_TARGET_DEFINITIONS Ops.td)
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/")

mlir_tablegen(Ops.h.inc -gen-op-decls "${TBLGEN_INCDIRS}")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "${TBLGEN_INCDIRS}")
add_public_tablegen_target(ArcDialectOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TypeInferenceOpInterface.td)
mlir_tablegen(TypeInferenceOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TypeInferenceOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ArcDialectTypeInferenceOpInterfaceIncGen)
