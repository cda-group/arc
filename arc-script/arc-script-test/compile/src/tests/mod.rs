#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

macro_rules! test {
    { $name:ident , $file:literal } => { mod $name { arc_script::include!($file); }}
}

mod expect_pass {
    test!(t00, "src/tests/expect_pass/basic_pipe.rs");
    test!(t01, "src/tests/expect_pass/binops.rs");
    test!(t02, "src/tests/expect_pass/enum_pattern.rs");
    test!(t03, "src/tests/expect_pass/enum_pattern_nested.rs");
    test!(t04, "src/tests/expect_pass/enums.rs");
    test!(t05, "src/tests/expect_pass/fib.rs");
    test!(t06, "src/tests/expect_pass/fun.rs");
    test!(t07, "src/tests/expect_pass/if_let.rs");
    test!(t08, "src/tests/expect_pass/ifs.rs");
    test!(t09, "src/tests/expect_pass/lambda.rs");
    test!(t10, "src/tests/expect_pass/literals.rs");
    test!(t11, "src/tests/expect_pass/nested_if.rs");
    test!(t12, "src/tests/expect_pass/option.rs");
    test!(t13, "src/tests/expect_pass/path.rs");
    test!(t14, "src/tests/expect_pass/pattern.rs");
    test!(t15, "src/tests/expect_pass/if.rs");
    test!(t16, "src/tests/expect_pass/structs.rs");
}

mod expect_mlir_fail_todo {
    test!(t00, "src/tests/expect_mlir_fail_todo/pipe.rs");
    test!(t01, "src/tests/expect_mlir_fail_todo/task_filter.rs");
    test!(t02, "src/tests/expect_mlir_fail_todo/task_id_untagged.rs");
    test!(t03, "src/tests/expect_mlir_fail_todo/task_map.rs");
    test!(t04, "src/tests/expect_mlir_fail_todo/task_with_funs.rs");
    test!(t05, "src/tests/expect_mlir_fail_todo/map_state.rs");
    test!(t06, "src/tests/expect_mlir_fail_todo/task_unique.rs");

    // Integration tests
    mod extern_fun;
    mod extern_state_update;
}
