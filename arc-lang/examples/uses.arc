# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
type Person = #{name: str, age: i32}

use Person as Human; # Creates an alias

def main(): Person {
    val person: Person = #{name:"Bob", age:35};
    val human: Human = Person;
}
# ANCHOR_END: example
