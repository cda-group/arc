<h1 align="center">Arc-Script - Dataflow Language</h1>

Arc-Script is a programming language for dataflow computation.
Dataflow programming is about approaching complicated problems by decomposing them into stages.
Each stage reads the output of the previous stage and forwards it to the next stage.
To this end, stages are both independent and pipelined, which means the program as a whole can
exploit pipeline parallelism. Stages are typically modeled declaratively with *combinators*, i.e.,
control-flow abstractions. Combinators describe the outputs of each stage is in terms of its
inputs while hiding details about how the output was derived. As indicated in its name, a combinator
is compositional and is often implemented as a higher-order function which may take other combinators
as parameters.

```
                                           +------> Map(|x| x + 1) ------+
---> Map(|x| x + 1) ---> Split(|x| x % 2) -+                             +-> Merge() --->
                                           +-> Scan(|agg, x| (agg + x)) -+
```

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
task Identity() i32 -> i32 {
    on event => emit event
}
```

In the above code, `Identity` is a task which takes an stream of `i32`s
on the task's input port and outputs a stream of `i32` on the task's output port.
On receiving an `event`, it is emitted without modification.

Tasks may also be initialized with parameters, such as the `Map` task.

```
task Map(f: fun(i32) -> i32) i32 -> i32 {
    on event => emit f(event)
}
```

The `state` keyword is used to declare and initialize a mutable state variable.
Each state variable is scoped within the task it is declared within and may not
escape to the outside.

```
task Scan(init: i32, f: fun(i32, i32) -> i32) i32 -> i32 {
    state agg: i32 = init;
    on event => {
        let agg' = f(agg, x) in
        agg := agg';
        emit agg'
    }
}
```

Tasks may have more than one input and output ports. As the arrival order of
events between ports is non-deterministic, events are tagged with a label that
indicates which port they are associated with. The concept of tagged ports is similar
to the concept of `enum`s (i.e., discriminated unions), but also different. An `enum` is a
sum type whose set of values ranges over the sum of its variants. The operation of
summation is commutative (order of operands is insignificant), i.e.:
```
VariantA(TypeA) + VariantB(TypeB) = VariantB(TypeA) + VariantA(TypeB)
```
Tagged ports are in contrast not associative:
```
PortA(TypeA) + PortB(TypeA) ≠ PortB(TypeB) + PortA(TypeA)
```
The reason why this is the case is because the order in which ports are
declared corresponds to the order in which streams connect to the operator.

```
task Tee() i32 -> (A(i32), B(i32)) {
    on event => {
        emit A(event);
        emit B(event)
    }
}

task Merge() (A(i32), B(i32)) -> i32 {
    on {
        A(event) => emit event,
        B(event) => emit event,
    }
}

task Flip() (A(i32), B(i32)) -> (C(i32), D(i32)) {
    on {
        A(event) => emit D(event),
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

Pipelines are nothing more than abstractions over streams, and can therefore be composed as well.

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

# [TODO] Control-Flow

```
task Looper() <In(i32)> -> <Out(i32)> {
    port Loop(i32);
    on {
      In(event) => emit Loop(event)
    }
}
```

## Compilation

Arc-Script programs are compiled in three stages.

As one might have noticed, tasks and functions both accept two lists of parameters.
The first list contains *eager* values which are known a-priori to deploying the streaming pipeline.
The second list contains *lazy* values which are only known after deploying the pipeline.
All values except for streams are *eager*.

In the first stage of compilation, the arc-script program is evaluated into a logical dataflow graph of tasks connected by streams.
The logical dataflow graph is a program represented as a graph of logical tasks connected by channels.


