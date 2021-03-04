#![allow(unused)]

#[cfg(test)]
mod insta;

// #[cfg(test)]
// mod trybuild;

// #[cfg(test)]
// mod compiletest;

mod tests {
    mod expect_pass {
        mod basic_pipe;
        mod binops;
        mod enum_pattern;
        mod enum_pattern_nested;
        mod enums;
        mod fib;
        mod fun;
        mod if_let;
        mod ifs;
        mod lambda;
        mod literals;
        mod nested_if;
        mod option;
        mod path;
        mod pattern;
        mod r#if;
        mod structs;
    }
    mod expect_mlir_fail_todo {
        mod extern_fun;
        mod pipe;
        mod stateful;
        mod task_filter;
        mod task_identity_untagged;
        mod task_map;
        mod task_with_funs;
    }
}
