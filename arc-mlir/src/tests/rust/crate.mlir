// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

"rust.crate"() ( {
"rust.func"() ( {

}) {sym_name = "the-function-name-0", type = () -> () } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)

}) {sym_name = "the-function-name-1", type = (!rust<"f32">) -> !rust<"f32"> } : () -> ()

"rust.func"() ( {
 ^bb0(%arg0: !rust<"f64">):
 "rust.return"(%arg0) : (!rust<"f64">) -> (!rust<"f64">)

}) {sym_name = "the-function-name-2", type = (!rust<"f64">) -> !rust<"f64"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "name-of-the-crate-0" }: () -> ()

// -----

"rust.crate"() ( {

// expected-error@+2 {{'rust.func' op requires a type attribute 'type'}}
// expected-note@+1 {{see current operation: "rust.func"() (}}
"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)
}) {sym_name = "the-function-name-1" } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "name-of-the-crate-1" }: () -> ()

// -----

"rust.crate"() ( {
// expected-error@+2 {{'rust.func' op expected 1 arguments to body region, found 2}}
// expected-note@+1 {{see current operation: "rust.func"() (}}
"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">, %arg1: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)
}) {sym_name = "the-function-name-1", type = (!rust<"f32">) -> !rust<"f32"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "name-of-the-crate-2" }: () -> ()

// -----

"rust.crate"() ( {
// expected-error@+2 {{'rust.func' op expected body region argument #0 to be of type '!rust.f64', found '!rust.f32'}}
// expected-note@+1 {{see current operation: "rust.func"() (}}
"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)
}) {sym_name = "the-function-name-1", type = (!rust<"f64">) -> !rust<"f32"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "name-of-the-crate-3" }: () -> ()

// -----

"rust.crate"() ( {
"rust.func"() ( {
 ^bb0(%arg0: !rust<"f32">):
// expected-error@+2 {{'rust.return' op result type does not match the type of the function: expected '!rust.f64' but found '!rust.f32'}}
// expected-note@+1 {{see current operation: %0 = "rust.return"}}
 "rust.return"(%arg0) : (!rust<"f32">) -> (!rust<"f32">)
}) {sym_name = "the-function-name-1", type = (!rust<"f32">) -> !rust<"f64"> } : () -> ()

"rust.crate_end"() : () -> ()
} ) { sym_name = "name-of-the-crate-3" }: () -> ()
