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
    on A(event) => emit B(event)
}
```

In the above code, `Identity` is a task which takes an stream of `i32`s on an input port `A`, and outputs a stream of `i32`s on output port `B`.
On receiving an `event` on port `A`, the event is emitted to port `B` without modification.

Tasks may also be initialized with parameters, such as the `Map` task.

```
task Map(f: fun(i32) -> i32) (A(i32)) -> (B(i32)) {
    on A(event) => emit B(f(event))
}
```

The `state` keyword is used to declare and initialize a mutable state variable.

```
task Scan(init: i32, f: fun(i32, i32) -> i32) (A(i32)) -> (B(i32)) {
    state agg: i32 = init;
    on A(event) => {
        let agg' = f(agg, x) in
        agg := agg';
        emit B(agg')
    }
}
```

Tasks can have multiple inputs and outputs:

```
task Tee() (A(i32)) -> (B(i32), C(i32)) {
    on A(event) => {
        emit B(event);
        emit C(event)
    }
}

task Merge() (A(i32), B(i32)) -> (C(i32)) {
    on {
        A(event) => emit C(event),
        B(event) => emit C(event),
    }
}
```

Tasks can then be instantiated and connected.

```
fun pipeline() (stream0: ~i32) -> ~i32 {
    let stream1 = Identity() (stream0) in
    let (stream2, stream3) = Tee() (stream1) in
    let stream4 = Map(|x| x + 1) (stream2) in
    let stream5 = Scan(|agg, x| (agg + x)) (stream3) in
    let stream6 = Merge() (stream4, stream5) in
    stream6
}
```

The produced dataflow graph looks like:

```
                            +------> Map(|x| x + 1) ------+
---> Identity() ---> Tee() -+                             +-> Merge() --->
                            +-> Scan(|agg, x| (agg + x)) -+
```

Using the piping operator, `|>`, tasks can be chained together in more readable syntax.

```
fun pipeline() (stream: ~i32) -> ~i32 {
    stream
        |> Identity()
        |> Tee()
        |> (Map(|x| x + 1), Scan(|agg, x| (agg + x, agg + x)))
        |> Merge()
}
```

Pipelines are nothing more than abstractions over streams, and can therefore be composed:

```
fun pipeline_3() (stream: ~i32) -> ~i32 {
    stream
        |> pipeline()
        |> pipeline()
        |> pipeline()
}
```

Pipelines can also accept values, such as:

```
fun pipeline_n(n: i32) (stream: ~i32) -> ~i32 {
    if n > 0 {
        stream |> pipeline() |> pipeline_n(n-1)
    } else {
        stream
    }
}
```

## Compilation

Arc-Script programs are compiled in three stages.
As one might have noticed, `task`s and `fun`ctions both accept two lists of parameters.
The first list contains *eager* values which are known a-priori to deploying the streaming pipeline.
The second list contains *lazy* values which are only known after deploying the pipeline.
All values except for streams are *eager*.

In the first stage of compilation, the arc-script program is evaluated into a logical dataflow graph of tasks connected by streams.
The logical dataflow graph is a program represented as a graph of logical tasks connected by channels.


