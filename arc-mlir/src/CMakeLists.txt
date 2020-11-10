set(ARC_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../llvm-project/mlir/include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../mlir/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

# Ugly hack to try to detect the jar of the arc scala tools in order
# to make wrapper shell script
file(GLOB ARC_JAR ${PROJECT_SOURCE_DIR}/../../../arc-frontend/target/scala-*/Arc-assembly-*.jar)
message(STATUS "ARC_JAR => ${ARC_JAR}")

configure_file(
    ${ARC_MLIR_SOURCE_DIR}/arc-to-mlir.in
    ${LLVM_BINARY_DIR}/bin/arc-to-mlir
    @ONLY
    )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(corrosion)

corrosion_import_crate(MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../arc-script/core/Cargo.toml")
