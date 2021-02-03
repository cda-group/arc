use arc_script_bridge::arc_script;

#[arc_script("basic.arc")]
mod script {}

fn main() {
    assert_eq!(1, script::test());
}
