set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(arc-mlir main.cpp)
llvm_update_compile_flags(arc-mlir)
target_link_libraries(arc-mlir PRIVATE LLVMSupport)
