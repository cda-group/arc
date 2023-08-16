set(ARC_MLIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(../llvm-project/mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(ARC_LANG_EXE ${LLVM_BINARY_DIR}/bin/arc-lang)

set(ENV{CARGO_HTTP_DEBUG} "true")

add_subdirectory(corrosion)

get_filename_component(ARC_CARGO_TARGET_DIR "../../cargo/build"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_CARGO_BIN "../../bin"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_TOOLS_BIN_DIR "../../bin"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_LANG_DIR "../../arc-lang"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

get_filename_component(ARC_LANG_TARGET_DIR "../../arc-lang/target"
  REALPATH BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE)

get_filename_component(ARC_RUNTIME_DIR "../../arc-lang/crates/runtime/"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

get_filename_component(ARC_MLIR_DIR "../"
  REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE)

add_custom_target(arc-lang ALL
  COMMAND cargo build --manifest-path=${ARC_LANG_DIR}/Cargo.toml --release --target-dir=${ARC_LANG_TARGET_DIR}
  COMMAND mv -f ${ARC_LANG_TARGET_DIR}/release/arc-lang ${ARC_LANG_EXE}
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

add_custom_target(arc-lang-check DEPENDS check-arc-mlir
  COMMAND cd ${ARC_LANG_DIR}; ${ARC_CARGO_BIN}/arc-cargo test
)

install(DIRECTORY ${ARC_LANG_DIR}/stdlib DESTINATION share/arc/)
install(DIRECTORY ${ARC_RUNTIME_DIR} DESTINATION share/arc/)
install(PROGRAMS ${LLVM_TOOLS_BINARY_DIR}/arc DESTINATION bin)
install(PROGRAMS ${LLVM_TOOLS_BINARY_DIR}/arc-lang DESTINATION bin)
