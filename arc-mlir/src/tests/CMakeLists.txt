# Required for the lit site config to specify the path to the
# arc-lang tool.
set(ARC_LANG_TOOLS_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

# Stolen from the mlir repo

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ARC_MLIR_TEST_DEPENDS
  FileCheck count not
  arc-mlir dune-build
)

add_lit_testsuite(check-arc-mlir "Running the ARC-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_TEST_DEPENDS} ${ARC_MLIR_TEST_DEPENDS}
)

set_target_properties(check-arc-mlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(ARC_MLIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ARC_MLIR_TEST_DEPENDS}
)
