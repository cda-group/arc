set(ARC_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../llvm-project/mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS arc-opts.td)
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/mlir/include/")
mlir_tablegen(arc-opts.inc -gen-rewriters ${TBLGEN_INCDIRS} "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ArcDialectOptsIncGen)


set(ARCMLIR_MODULES main.cpp arc-dialect.cpp arc-opts.cpp)

set(LIBS
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRLoopsToGPU
  MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRLinalgOps
  MLIRLLVMIR
  MLIRLoopOps
  MLIRNVVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRROCDLIR
  MLIRSPIRV
  MLIRStandardToSPIRVTransforms
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
)

add_llvm_executable(arc-mlir ${ARCMLIR_MODULES})
add_dependencies(arc-mlir ArcDialectOpsIncGen)
add_dependencies(arc-mlir ArcDialectOptsIncGen)

llvm_update_compile_flags(arc-mlir)

whole_archive_link(arc-mlir ${LIBS})

target_link_libraries(arc-mlir PRIVATE ${LIBS} LLVMSupport)

# Ugly hack to try to detect the jar of the arc scala tools in order
# to make wrapper shell script
file(GLOB ARC_JAR ${PROJECT_SOURCE_DIR}/../../../target/scala-*/Arc-assembly-*.jar)
message(STATUS "ARC_JAR => ${ARC_JAR}")

configure_file(
    ${ARC_MLIR_SOURCE_DIR}/arc-to-mlir.in
    ${LLVM_BINARY_DIR}/bin/arc-to-mlir
    @ONLY
    )

add_subdirectory(tests)
add_subdirectory(include)
