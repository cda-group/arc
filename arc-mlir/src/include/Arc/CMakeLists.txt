set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/)
set(MLIR_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvm-project/mlir/include/
)

add_mlir_dialect(Arc arc Arc)
