set(ARC_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../tensorflow-mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(arc-mlir main.cpp arc-dialect.cpp)
llvm_update_compile_flags(arc-mlir)
add_dependencies(arc-mlir ArcDialectOpsIncGen)

target_link_libraries(arc-mlir PRIVATE
  LLVMSupport
  MLIRAnalysis
  MLIRIR
  MLIRTransforms
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
  MLIROptMain
  MLIRParser
)

whole_archive_link(arc-mlir
  MLIRAffineOps
  MLIRStandardOps)

# Ugly hack to try to detect the jar of the arc scala tools in order
# to make wrapper shell script
file(GLOB ARC_JAR ${PROJECT_SOURCE_DIR}/../../../arc-frontend/target/scala-*/Arc-assembly-*.jar)
message(STATUS "ARC_JAR => ${ARC_JAR}")

configure_file(
    ${ARC_MLIR_SOURCE_DIR}/arc-to-mlir.in
    ${LLVM_BINARY_DIR}/bin/arc-to-mlir
    @ONLY
    )

add_subdirectory(tests)
add_subdirectory(include)
