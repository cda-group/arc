use arc_script_bridge::arc_script;

#[arc_script("script.arc")]
mod script { }

#[test]
fn test() {
    assert_eq!(1, script::test());
}
