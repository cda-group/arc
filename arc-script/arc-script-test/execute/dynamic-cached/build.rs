// use arc_script::{Field, Fun, Script};
use arc_script_build::Builder;

/// Test to see if `Script` works as intended.
fn main() {
    // This pre-builds any file in the crate whose filename is `main.arc`.
    // Pre-building improves performance by caching build-artifacts, but
    // does not do much else.
    Builder::default().build();
    println!("cargo:rerun-if-env-changed=ARCSCRIPT_MLIR_BACKEND");
    // arc_script!("./script.arc")
    //     .compile_fun1(5i8, "foo");
    //     .compile_fun2(0i32, 0u8);
    //     .build();

//     Script::new("src/script.arc")
//         .stage(
//             Fun::new("fun1")
//                 .arg("arg1", 5i8)
//                 .arg("arg2", "foo")
//                 .arg("arg3", "bar")
//                 .arg(
//                     "argx",
//                     (
//                         Field::new("foo", 3),
//                         Field::new("foo", 3),
//                         Field::new("3", 5),
//                         Field::new("3", 5),
//                     ),
//                 ),
//         )
//         .stage(
//             Fun::new("fun2")
//                 .arg("arg1", 0)
//                 .arg("argz", 0i8)
//                 .arg("baz", (("foo", 0i8), 0i8)),
//         )
//         .compile();
}
