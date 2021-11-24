set(ARC_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../llvm-project/mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

set(ENV{CARGO_HTTP_DEBUG} "true")

add_subdirectory(corrosion)

get_filename_component(ARC_CARGO_TARGET_DIR "../../cargo/build"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_CARGO_BIN "../../bin"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_TOOLS_BIN_DIR "../../bin"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_LANG_SRC_DIR "../../arc-lang"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

get_filename_component(ARC_CODEGEN_SRC_DIR "../../arc-codegen"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

get_filename_component(ARC_RUNTIME_SRC_DIR "../../arc-runtime"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

get_filename_component(ARC_MLIR_SRC_DIR "../"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

get_filename_component(ARC_CARGO_BIN "../../bin"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

add_custom_target(arc-codegen-check DEPENDS check-arc-mlir
  COMMAND cd ${ARC_SCRIPT_SRC_DIR}; ${ARC_CARGO_BIN}/arc-cargo test --package=arc-codegen
)

add_custom_target(arc-runtime-check DEPENDS check-arc-mlir
  COMMAND cd ${ARC_SCRIPT_SRC_DIR}; ${ARC_CARGO_BIN}/arc-cargo test --package=arc-runtime
)
