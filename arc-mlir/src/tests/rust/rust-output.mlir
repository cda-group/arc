// RUN: arc-mlir -crate %t %s && cargo build -j 1 --manifest-path=%t/this_is_the_name_of_the_crate/Cargo.toml

"rust.crate"() ( {

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)

}) {sym_name = "this_is_the_name_of_the_second_function", type = (!rust<"f32">) -> !rust<"f32"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f64">, %arg1: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_third_function", type = (!rust<"f64">, !rust<"f32">) -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0:
 %r = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 "rust.return"(%r) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_fourth_function", type = () -> !rust<"f64"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "this_is_the_name_of_the_crate" }: () -> ()
