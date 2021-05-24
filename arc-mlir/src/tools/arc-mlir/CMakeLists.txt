get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRLoopAnalysis
  MLIRAffineTransformsTestPasses
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSPIRVTestPasses
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

add_llvm_tool(arc-mlir
  ArcOptMain.cpp
  Main.cpp
)
llvm_update_compile_flags(arc-mlir)
target_link_libraries(arc-mlir
  PRIVATE ArcDialect RustDialect ${LIBS} LLVMSupport)
