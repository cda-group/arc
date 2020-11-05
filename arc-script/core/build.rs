extern crate lalrpop;

fn main() {
    compile_parser();
    #[cfg(feature = "mlir")]
    compile_mlir();
}

fn compile_parser() {
    lalrpop::Configuration::new()
        .emit_whitespace(false)
        .use_cargo_dir_conventions()
        .process()
        .unwrap();
}

#[cfg(feature = "mlir")]
fn compile_mlir() {
    use fstrings::*;
    use std::env;
    let arc_script = env::var("CARGO_MANIFEST_DIR").unwrap();
    let arc_mlir = f!("{arc_script}/../../arc-mlir");
//     std::fs::create_dir(f!("{arc_mlir}/build/llvm",))

    cmake::Config::new(f!("{arc_mlir}/llvm-project/llvm"))
        .env("CXX", "clang++")
        .env("CC", "clang")
        .define("CMAKE_GENERATOR", "Ninja")
        .define("CMAKE_INSTALL_PREFIX", f!("{arc_mlir}/build/llvm-install"))
        .define("CMAKE_EXPORT_COMPILE_COMMANDS", "1")
        .define("CMAKE_BUILD_TYPE", "Debug")
        .define("LLVM_TARGETS_TO_BUILD", "host")
        .define("LLVM_USE_LINKER", "gold")
        .define("LLVM_ENABLE_ASSERTIONS", "ON")
        .define("BUILD_SHARED_LIBS", "0")
        .define("LLVM_USE_SPLIT_DWARF", "ON")
        .define("LLVM_CCACHE_BUILD", "1")
        .define("LLVM_BUILD_EXAMPLES", "OFF")
        .define("LLVM_ENABLE_PROJECTS", "mlir")
        .define("LLVM_EXTERNAL_PROJECTS", "arc-mlir")
        .define("LLVM_EXTERNAL_ARC_MLIR_SOURCE_DIR", f!("{arc_mlir}/src"))
        .define("LLVM_HAVE_TF_API", "OFF")
        .build();

    cmake::Config::new(f!("{arc_mlir}/build/llvm-project/llvm"))
        .build_arg(f!("{arc_mlir}/llvm-build"))
        .build();
}
