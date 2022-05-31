# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
type Person = #{name: str, age: i32};

use Person as Human; # Creates an alias

def main(): Person {
    val person: Person = #{name:"Bob", age:35};
    val human: Human = person;
}
# ANCHOR_END: example
