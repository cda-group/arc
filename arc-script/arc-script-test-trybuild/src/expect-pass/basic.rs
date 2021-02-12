use arc_script;

#[arc_script::compile("basic.arc")]
mod script {}

fn main() {
    assert_eq!(1, script::test());
}
