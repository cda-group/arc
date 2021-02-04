<h1 align="center">Arc-Script - Dataflow Language</h1>

Arc-Script is a programming language for dataflow computation.

The core-abstractions of Arc-Script programs are tasks, streams, events, and pipelines.
A *task* is a concurrent primitive which incrementally transduces streams of data.
A *stream* is an unbounded sequence of events.
An *event* is a value discriminated by time. Events in a stream are ordered by their time of arrival.
Streams connect to tasks through input and output ports.
Multiple streams can be multiplexed on the same port.
Tasks receive and process one event at one input port at a time, and may emit multiple events at multiple output ports for each input event.
Additionally, tasks may hold onto mutable *state* which is persisted over time.
The order in which events are received is not deterministic, but can be enforced through state.
A *pipeline* is a function which maps input streams to output streams.

Before going further, let's dive into some examples of how to program with `task`s and events in Arc-Script.

The identity task over integers can be programmed as:

```
task Identity() (A(i32)) -> (B(i32)) {
    on A(x) => emit B(x)
}
```

In the above code, `Identity` is a task which takes an stream of `i32`s on an input port `A`, and outputs a stream of `i32`s on output port `B`.
On receiving an element on port `A`, the element is emitted to port `B` without modification.

Tasks may also be initialized with parameters, such as the `Map` task.

```
task Map(f: fun(i32) -> i32) (A(i32)) -> (B(i32)) {
    on A(x) => emit B(f(x))
}
```

Tasks can then be instantiated and connected.

```
fun pipeline(source: ~i32) -> ~i32 {
    let output1 = Identity() (source);
    let y = Identity(output1) 
}
```

Arc-Script programs are lowered through staged evaluation into dataflow graphs. The graph is then 
