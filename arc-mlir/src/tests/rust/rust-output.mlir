// RUN: arc-mlir -crate %t %s && cargo build -j 1 --manifest-path=%t/this_is_the_name_of_the_crate/Cargo.toml

"rust.crate"() ( {
"rust.func"() ( {

}) {sym_name = "this_is_the_name_of_the_function", type = () -> () } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)

}) {sym_name = "the-function-name-1", type = (!rust<"f32">) -> !rust<"f32"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f64">):
 "rust.return"(%arg0) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "the-function-name-2", type = (!rust<"f64">) -> !rust<"f64"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "this_is_the_name_of_the_crate" }: () -> ()
