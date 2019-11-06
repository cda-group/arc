set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(arc-to-mlir main.cpp)
llvm_update_compile_flags(arc-to-mlir)
target_link_libraries(arc-to-mlir PRIVATE LLVMSupport)
