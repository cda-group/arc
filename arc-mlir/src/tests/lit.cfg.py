# -*- Python -*-
# Mostly stolen from the mlir test suite

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'ARC-MLIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.arc', '.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

from pathlib import Path

# /path/to/arc/arc-mlir/src/tests/lit.cfg.py => /path/to/arc/arc-lang/stdlib/
arc_stdlib_dir = Path(__file__).parents[3] / 'arc-lang' / 'stdlib' 

llvm_config.with_environment('ARC_LANG_STDLIB_PATH', arc_stdlib_dir / 'stdlib.arc')
llvm_config.with_environment('ARC_MLIR_STDLIB_PATH', arc_stdlib_dir / 'stdlib.mlir')

tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir,
             config.arcscript_tools_dir]
tools = [
    'mlir-opt',
    'mlir-tblgen',
    'mlir-translate',
    'arc-mlir',
    'arc-lang',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
