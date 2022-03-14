# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
val a0 = "hello";

assert(a0.str_eq("hello"));
assert(not a0.str_eq("world"));

val a1 = i32_to_string(1);
assert(a1.str_eq("1"));

val a2 = "world";
a2.push_char('!');
assert(a2.str_eq("world!"));

a2.insert_char(0u32, '1');
assert(str_eq(a2, "!world!"));

val a3 = "";
assert(not a2.is_empty_str());
assert(a3.is_empty_str());

val a4 = "hey";
a4.clear_str();
assert(is_empty_str(a4));

val a5 = "(((";
val a6 = ")))";
assert(a5.concat(a6).str_eq("((()))"));
# ANCHOR_END: example
}
