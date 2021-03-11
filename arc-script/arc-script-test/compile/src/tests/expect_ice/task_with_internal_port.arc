task Delay() (Input(~i32)) -> (Output(~i32)) {
    port Trigger(~i32)

    on {
        Source::Input(event) => emit Sink::Trigger(event) after 10s,
        Source::Trigger(event) => emit Sink::Output(event)
    }
}
