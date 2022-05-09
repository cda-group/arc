# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def wordcount(lines) =
    from line in lines,
         word in line.split(" ")
    group word
    window 10m every 5m
    compute count
# ANCHOR_END: example

def main() {
# ANCHOR: polymorphic
val df = DataFrame::read("/path/to/data.csv");
val wc0 = wordcount(df);

val s = DataStream::read("localhost:8080");
val wc1 = wordcount(s);
# ANCHOR_END: polymorphic
}
