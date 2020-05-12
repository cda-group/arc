// RUN: arc-mlir -crate %t %s && cargo build -j 1 --manifest-path=%t/this_is_the_name_of_the_crate/Cargo.toml

module @"this_is_the_name_of_the_crate" {

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)

}) {sym_name = "this_is_the_name_of_the_second_function", type = (!rust<"f32">) -> !rust<"f32"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f64">):
 "rust.return"(%arg0) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_third_function", type = (!rust<"f64">) -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0:
 %r = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 "rust.return"(%r) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_fourth_function", type = () -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0:
 %r = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 %x = "rust.unaryop"(%r) {op="-"} : (!rust<"f64">) -> (!rust<"f64">)
 "rust.return"(%x) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_fifth_function", type = () -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f64">):
 %r = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 %x = "rust.unaryop"(%r) {op="-"} : (!rust<"f64">) -> (!rust<"f64">)
 %y = "rust.binaryop"(%arg0, %x) {op="+"} : (!rust<"f64">, !rust<"f64">) -> (!rust<"f64">)
 "rust.return"(%y) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_sixth_function", type = (!rust<"f64">) -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"bool">, %arg1: !rust<"f64">):
   %r = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
   %all = "rust.if"(%arg0) ( {
       	%x = "rust.unaryop"(%r) {op="-"} : (!rust<"f64">) -> (!rust<"f64">)
       	%y = "rust.binaryop"(%arg1, %x) {op="+"} : (!rust<"f64">, !rust<"f64">) -> (!rust<"f64">)
       	"rust.block.result"(%y) : (!rust<"f64">) -> !rust<"f64">
       },  {
       	"rust.block.result"(%r) : (!rust<"f64">) -> !rust<"f64">
       }) : (!rust<"bool">) -> !rust<"f64">
 "rust.return"(%all) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "this_is_the_name_of_the_seventh_function", type = (!rust<"bool">, !rust<"f64">) -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0(%a: !rust<"f64">, %b: !rust<"f64">):
 %x = "rust.compop"(%a, %b) {op="<"} : (!rust<"f64">, !rust<"f64">) -> (!rust<"bool">)
 "rust.return"(%x) : (!rust<"bool">) -> (!rust<"bool">)

}) {sym_name = "this_is_the_name_of_the_eigth_function", type = (!rust<"f64">, !rust<"f64">) -> !rust<"bool"> } : () -> ()

"rust.func"() ( {
 ^bb0:
 %a = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 %r = "rust.method_call"(%a) {method="sin"} : (!rust<"f64">) -> (!rust<"f64">)
 "rust.return"(%r) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "testing_a_method_call", type = () -> !rust<"f64"> } : () -> ()

"rust.func"() ( {
 ^bb0:
 %a = "rust.constant"() {value="3.14"} : () -> (!rust<"f64">)
 %b = "rust.constant"() {value="2.0"} : () -> (!rust<"f64">)
 %r = "rust.method_call"(%a, %b) {method="log"} : (!rust<"f64">, !rust<"f64">) -> (!rust<"f64">)
 "rust.return"(%r) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "testing_a_method_call_with_args", type = () -> !rust<"f64"> } : () -> ()

}

