set(ARC_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../llvm-project/mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS include/arc/Opts.td)
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(TBLGEN_INCDIRS "-I${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/mlir/include/")
mlir_tablegen(Opts.inc -gen-rewriters ${TBLGEN_INCDIRS} "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ArcDialectOptsIncGen)

set(ARCMLIR_MODULES Main.cpp lib/arc/Dialect.cpp lib/arc/Opts.cpp lib/arc/Types.cpp lib/ArcOptMain.cpp)

set(RUSTMLIR_MODULES lib/rust/Dialect.cpp lib/rust/Types.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopOpsTransforms
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSPIRV
  MLIRSPIRVTestPasses
  MLIRSPIRVTransforms
  MLIRTransforms
  MLIRTransformUtils
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser
)

add_llvm_executable(arc-mlir ${ARCMLIR_MODULES} ${RUSTMLIR_MODULES})
add_dependencies(arc-mlir ArcDialectOpsIncGen)
add_dependencies(arc-mlir ArcDialectOptsIncGen)

add_dependencies(arc-mlir RustDialectOpsIncGen)

llvm_update_compile_flags(arc-mlir)

target_link_libraries(arc-mlir PRIVATE ${LIBS} LLVMSupport)

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
