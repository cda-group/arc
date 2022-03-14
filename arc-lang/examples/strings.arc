# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def main() {
    val a0 = "hello";

    assert(str_eq(a0, "hello"));
#     assert(str_eq(a0, "hello"));
#
#     val a1 = i32_to_string(1);
#     assert(str_eq(a1, "1"));
#
#     val a2 = "world";
#     push_char(a2, '!');
#     assert(str_eq(a2, "world!"));
#
#     insert_char(a2, 0, '1');
#     assert(str_eq(a2, "!world!"));
#
#     val a3 = "";
#     assert(not is_empty_str(a2));
#     assert(is_empty_str(a3));
#
#     val a4 = "hey";
#     clear_str(a4);
#     assert(is_empty_str(a4));
#
#     val a5 = "(((";
#     val a6 = ")))";
#     assert(str_eq(concat(a5, a6), "((()))"));
}
