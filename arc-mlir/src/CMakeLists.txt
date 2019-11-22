include_directories(../tensorflow-mlir/include/)
set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(arc-mlir main.cpp)
llvm_update_compile_flags(arc-mlir)
target_link_libraries(arc-mlir PRIVATE
  LLVMSupport
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRTransforms)

whole_archive_link(arc-mlir
  MLIRAffineOps
  MLIRStandardOps)

# Ugly hack to try to detect the jar of the arc scala tools in order
# to make wrapper shell script
file(GLOB ARC_JAR ${PROJECT_SOURCE_DIR}/../../../target/scala-*/Arc-assembly-*.jar)
message(STATUS "ARC_JAR => ${ARC_JAR}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/arc-to-mlir.in
    ${LLVM_BINARY_DIR}/bin/arc-to-mlir
    @ONLY
    )
