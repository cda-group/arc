# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def coerce(x: #{}): #{} {
    x
}

def main() {
    coerce(#{y:5});
}
