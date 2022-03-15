# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# The declaration order of top-level definitions is insignificant. In other words,
# functions can reference other functions declared farther down in the code.

def is_even(n) = if n == 0 { true } else { is_odd(n-1) }

def is_odd(n) = if n == 0 { false } else { is_even(n-1) }
# ANCHOR_END: example

def main() = {
  is_even(10);
}
